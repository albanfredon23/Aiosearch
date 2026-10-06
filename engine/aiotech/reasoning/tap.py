"""
TAP : évaluation des trajectoires et classement des réponses.

Pour chaque trajectoire (une interprétation de la requête), les candidats sont tirés des
preuves de cette trajectoire :
    entités     produits, lieux, personnes... décrits par des affirmations chiffrées
                (recommandations, comparaisons, requêtes à contraintes) ;
    valeurs     valeurs concurrentes d'un attribut de l'entité visée (« délai de
                rétractation : 14 jours ou 7 jours ? ») ;
    énoncés     phrases de source les plus admissibles au sens de l'ARG (repli général).

Chaque candidat passe la porte de Gödel du SCG (scg.py), puis reçoit un score TAP :
    appui        = 1 − exp(−Σ fiabilités des sources distinctes / 2)
    fidélité     = part des affirmations [FAIT] ou [INFÉRENCE] ([INCERTAIN] compte 1/2)
    cohérence    = support de chaîne ARG de la réponse rédigée (1 pour une réponse gabarit,
                   construite uniquement à partir de valeurs vérifiées)
    pénalité     = 0,25 par contradiction ouverte, plafonnée à 0,75
    qualité      = moyenne(cohérence, appui, fidélité) · (1 − pénalité)
    fiabilité    = T_G(A, qualité) = min(A, qualité)

Les 2 ou 3 réponses finales sont classées par fiabilité (en pourcentage entier), puis,
à fiabilité égale seulement, par le taux de choix appris des clics.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from aiotech.core.arg import ReachabilityGate
from aiotech.core.text import content_words, jaccard, split_sentences
from aiotech.core.units import satisfies
from aiotech.graph.entities import entity_key
from aiotech.graph.knowledge_graph import KnowledgeGraph, noisy_or
from aiotech.models import (
    Candidate,
    Claim,
    IntentPlan,
    Interpretation,
    Passage,
    Possibility,
    Premise,
    ScoredPassage,
    Source,
    Status,
    TapScore,
    Trajectory,
    ValueGroup,
    VerifiedClaim,
)
from aiotech.reasoning.scg import (
    UNKNOWN,
    ConstraintCheck,
    Requirement,
    check_constraint,
    facet_degree,
    godel_gate,
    kleene_dienes,
)
from aiotech.reasoning.template import entity_answer, value_answer
from aiotech.verification.engine import VerificationEngine, attribute_label, group_value

DraftKind = Literal["entity", "value", "statement"]

STATEMENT_FLOOR = 0.12
MAX_ENTITY_DRAFTS = 8
MAX_VALUE_DRAFTS = 6
MAX_STATEMENT_DRAFTS = 3
MAX_ATTRIBUTES = 5
_STATUS_WEIGHT = {Status.FAIT: 1.0, Status.INFERENCE: 1.0, Status.INCERTAIN: 0.5, Status.NON_VERIFIE: 0.0}


@dataclass(frozen=True)
class TrajectoryEvidence:
    interpretation: Interpretation
    passages: tuple[ScoredPassage, ...]
    claims: tuple[Claim, ...]

    def relevance(self) -> dict[str, float]:
        """Score hybride de chaque passage rapporté au meilleur passage de la trajectoire."""
        if not self.passages:
            return {}
        top = max(sp.score for sp in self.passages) or 1.0
        return {sp.passage.id: round(sp.score / top, 4) for sp in self.passages}


@dataclass(frozen=True)
class Draft:
    key: str
    label: str
    kind: DraftKind
    passage_ids: tuple[str, ...]
    entity_key: str | None = None
    attribute: str | None = None
    group: ValueGroup | None = None
    sentence: str | None = None
    sentence_passage: str | None = None
    corroborating: tuple[Source, ...] = ()
    arg_degree: float = 0.0


@dataclass
class Reasoner:
    plan: IntentPlan
    graph: KnowledgeGraph
    passages: Mapping[str, Passage]
    requirements: Sequence[Requirement] = ()
    verifier: VerificationEngine = field(default_factory=VerificationEngine)
    gate: ReachabilityGate = field(default_factory=ReachabilityGate)

    def __post_init__(self) -> None:
        self._open_conflicts: dict[str, list[str]] = {}
        for contradiction in self.graph.contradictions():
            if not contradiction.resolved:
                self._open_conflicts.setdefault(contradiction.entity_key, []).append(contradiction.attribute)
        self._holders = {k for e in self.plan.entities for k in self.graph.find_entities(entity_key(e))}
        self._query_stems = set(content_words(self.plan.query))

    def evaluate(self, evidence: TrajectoryEvidence) -> Trajectory:
        interpretation = evidence.interpretation
        if not evidence.passages:
            return Trajectory(interpretation_id=interpretation.id, admissible=False, best=None, candidates=(),
                              passages=0, rejection_reason="Aucune source pertinente pour cette lecture")
        relevance = evidence.relevance()
        candidates: list[Candidate] = []
        for family in self._families(evidence):
            evaluated = [self._candidate(d, evidence, relevance) for d in family]
            candidates.extend(evaluated)
            if any(not c.rejected for c in evaluated):
                break
        candidates.sort(key=lambda c: (not c.rejected, c.score.reliability), reverse=True)
        kept = [c for c in candidates if not c.rejected]
        best = kept[0] if kept else None
        reason = None
        if best is None:
            reason = candidates[0].rejection_reason if candidates else "Aucun candidat dans les sources trouvées"
        return Trajectory(
            interpretation_id=interpretation.id,
            admissible=best is not None,
            best=best,
            candidates=tuple(candidates),
            passages=len(evidence.passages),
            rejection_reason=reason,
        )

    def _families(self, evidence: TrajectoryEvidence) -> list[list[Draft]]:
        """Familles de candidats par ordre de priorité ; la suivante n'est essayée que si toutes sont rejetées."""
        entity_first = self.plan.question_type in ("recommendation", "comparison") or bool(self.plan.constraints)
        entities = self._entity_drafts(evidence)
        values = self._value_drafts(evidence)
        ordered = [entities, values] if entity_first else [values, entities]
        families = [family for family in ordered if family]
        statements = self._statement_drafts(evidence)
        if statements:
            families.append(statements)
        return families

    def _entity_drafts(self, evidence: TrajectoryEvidence) -> list[Draft]:
        comparison = self.plan.question_type == "comparison"
        relevance = evidence.relevance()
        by_entity: dict[str, list[Claim]] = {}
        for claim in evidence.claims:
            if claim.kind not in ("fact", "type"):
                continue
            key = self.graph.canonical_key(claim.subject_key)
            if not comparison and key in self._holders:
                continue
            if self._is_query_topic(key) and not comparison:
                continue
            by_entity.setdefault(key, []).append(claim)
        if comparison and self._holders:
            by_entity = {k: v for k, v in by_entity.items() if k in self._holders} or by_entity
        constrained = {c.attribute for c in self.plan.constraints} | {r.constraint.attribute for r in self.requirements}
        if constrained and not comparison:
            by_entity = {
                k: v for k, v in by_entity.items()
                if any(c.attribute in constrained for c in self.graph.claims_for(k)) or self._unit_match(k)
            }
        drafts = [
            Draft(
                key=f"entity:{key}",
                label=self.graph.entity_label(key),
                kind="entity",
                passage_ids=tuple(dict.fromkeys(c.passage_id for c in claims)),
                entity_key=key,
            )
            for key, claims in by_entity.items()
            if any(c.kind == "fact" for c in claims) or comparison
        ]
        drafts.sort(key=lambda d: max(relevance.get(p, 0.0) for p in d.passage_ids), reverse=True)
        return drafts[:MAX_ENTITY_DRAFTS]

    def _unit_match(self, key: str) -> bool:
        units = {c.quantity.unit for c in self.plan.constraints if c.attribute.startswith("unit:")}
        return bool(units) and any(c.quantity is not None and c.quantity.unit in units for c in self.graph.claims_for(key))

    def _is_query_topic(self, key: str) -> bool:
        stems = set(key.split())
        return bool(stems) and stems <= self._query_stems

    def _value_drafts(self, evidence: TrajectoryEvidence) -> list[Draft]:
        drafts: list[Draft] = []
        seen: set[tuple[str, str]] = set()
        for claim in evidence.claims:
            key = self.graph.canonical_key(claim.subject_key)
            if claim.kind != "fact" or not self._is_query_topic(key) or (key, claim.attribute) in seen:
                continue
            seen.add((key, claim.attribute))
            resolution = self.graph.resolve(key, claim.attribute)
            label = self.graph.entity_label(key)
            for group in resolution.groups:
                drafts.append(
                    Draft(
                        key=f"value:{key}:{claim.attribute}:{group_value(group)}",
                        label=f"{label} : {group_value(group)}",
                        kind="value",
                        passage_ids=tuple(dict.fromkeys(self.graph.claims[c].passage_id for c in group.claim_ids)),
                        entity_key=key,
                        attribute=claim.attribute,
                        group=group,
                    )
                )
        return drafts[:MAX_VALUE_DRAFTS]

    def _statement_drafts(self, evidence: TrajectoryEvidence) -> list[Draft]:
        sentences: list[tuple[str, Passage]] = []
        for sp in evidence.passages[:6]:
            for sentence in split_sentences(sp.passage.text):
                if 4 <= len(sentence.split()) <= 70:
                    sentences.append((sentence, sp.passage))
        if not sentences:
            return []
        rows = self.gate.admissibility(self.plan.query, [s for s, _ in sentences])
        ranked = sorted(
            ((rows[i][3], sentences[i][0], sentences[i][1]) for i in range(len(sentences))),
            key=lambda item: item[0], reverse=True,
        )
        best = ranked[0][0]
        if best < STATEMENT_FLOOR:
            return []
        chosen: list[tuple[float, str, Passage, list[Source]]] = []
        for adm, sentence, passage in ranked:
            if adm < STATEMENT_FLOOR:
                break
            words = set(content_words(sentence))
            duplicate = next((c for c in chosen if jaccard(words, set(content_words(c[1]))) >= 0.7), None)
            if duplicate is not None:
                if passage.source.id != duplicate[2].source.id:
                    duplicate[3].append(passage.source)
                continue
            if len(chosen) < MAX_STATEMENT_DRAFTS:
                chosen.append((adm, sentence, passage, []))
        return [
            Draft(
                key=f"statement:{hashlib.blake2b(sentence.encode(), digest_size=6).hexdigest()}",
                label=sentence if len(sentence) <= 90 else sentence[:87].rstrip() + "…",
                kind="statement",
                passage_ids=(passage.id,),
                sentence=sentence,
                sentence_passage=passage.id,
                corroborating=tuple(corroborating),
                arg_degree=round(min(1.0, adm / best), 4),
            )
            for adm, sentence, passage, corroborating in chosen
        ]

    def _sibling_terms(self, interpretation: Interpretation) -> tuple[str, ...]:
        return tuple(
            t for other in self.plan.interpretations if other.id != interpretation.id for t in other.facet_terms
        )

    def _candidate(self, draft: Draft, evidence: TrajectoryEvidence, relevance: Mapping[str, float]) -> Candidate:
        if draft.kind == "entity":
            premises, claims = self._entity_case(draft, evidence, relevance)
        elif draft.kind == "value":
            premises, claims = self._value_case(draft, evidence, relevance)
        else:
            premises, claims = self._statement_case(draft, evidence)
        gate = godel_gate(premises)
        open_conflicts = len(self._open_conflicts.get(draft.entity_key or "", []))
        if draft.kind == "statement" and any(c.status == Status.INCERTAIN for c in claims):
            open_conflicts = max(open_conflicts, 1)
        sources = {s.id: s.reliability for c in claims for s in c.sources}
        score = tap_score(gate.admissibility, claims, sources.values(), coherence=1.0, open_conflicts=open_conflicts)
        return Candidate(
            key=draft.key,
            label=draft.label,
            premises=tuple(premises),
            claims=tuple(claims),
            score=score,
            rejected=gate.rejected,
            rejection_reason=gate.reason,
        )

    def _facet_premise(self, text: str, interpretation: Interpretation, exclusive: bool = False) -> Premise | None:
        """`exclusive` : un candidat qui relève d'une facette voisine est hors sujet (produits, lieux...)."""
        siblings = self._sibling_terms(interpretation) if exclusive else ()
        degree = facet_degree(text, interpretation.facet_terms, siblings)
        if degree is None:
            return None
        return Premise(label=f"Relève de « {interpretation.label} »", degree=degree, kind="facet")

    def _entity_case(
        self, draft: Draft, evidence: TrajectoryEvidence, relevance: Mapping[str, float]
    ) -> tuple[list[Premise], list[VerifiedClaim]]:
        assert draft.entity_key is not None
        key = draft.entity_key
        all_claims = self.graph.claims_for(key, kinds=("fact", "type"))
        premises = [
            Premise(label="Pertinence des passages (rang hybride)",
                    degree=max(relevance.get(p, 0.0) for p in draft.passage_ids), kind="relevance"),
            Premise(label="Appui des sources (OU bruité)",
                    degree=round(noisy_or({c.source.id: c.source.reliability for c in all_claims}.values()), 4),
                    kind="evidence"),
        ]
        facet_text = " ".join([draft.label, *(c.quote for c in all_claims), *(c.value_text for c in all_claims)])
        facet = self._facet_premise(facet_text, evidence.interpretation, exclusive=True)
        if facet is not None:
            premises.append(facet)

        verified: dict[str, VerifiedClaim] = {}
        checks: list[tuple[ConstraintCheck, Requirement | None]] = []
        for constraint in self.plan.constraints:
            check = check_constraint(self.graph, key, constraint)
            checks.append((check, None))
            premises.append(Premise(label=check.describe(), degree=check.satisfaction, kind="constraint"))
        for requirement in self.requirements:
            check = check_constraint(self.graph, key, requirement.constraint)
            checks.append((check, requirement))
            degree = round(kleene_dienes(requirement.confidence, check.satisfaction), 4)
            premises.append(Premise(label=check.describe(), degree=degree, kind="constraint"))
        open_attrs = self._open_conflicts.get(key, [])
        premises.append(
            Premise(
                label="Contradiction ouverte : " + ", ".join(attribute_label(a) for a in open_attrs)
                if open_attrs else "Aucune contradiction ouverte",
                degree=UNKNOWN if open_attrs else 1.0,
                kind="consistency",
            )
        )

        ordered = [c.constraint.attribute for c, _ in checks if not c.constraint.attribute.startswith("unit:")]
        ordered += [c.attribute for c in all_claims if c.kind == "fact"]
        for attribute in [*dict.fromkeys(ordered), "type"]:
            if attribute in verified or len(verified) >= MAX_ATTRIBUTES + 1:
                continue
            claim = self.verifier.verify_attribute(self.graph, key, attribute, self.passages)
            if claim is not None:
                verified[attribute] = claim
        claims = list(verified.values())
        for check, source_requirement in checks:
            claims.append(self._constraint_claim(key, check, source_requirement, verified))
        return premises, claims

    def _constraint_claim(
        self, key: str, check: ConstraintCheck, requirement: Requirement | None, verified: Mapping[str, VerifiedClaim]
    ) -> VerifiedClaim:
        if check.outcome == "unknown":
            return VerifiedClaim(text=f"{check.constraint.label} : non vérifiable", status=Status.NON_VERIFIE,
                                 reason="Aucune source ne donne cette valeur pour ce candidat")
        premises = [v for a, v in verified.items() if a == check.constraint.attribute]
        if requirement is not None:
            req_claims = [self.graph.claims[c] for c in requirement.claim_ids if c in self.graph.claims]
            premises.append(self.verifier.verify_claim_set(requirement.constraint.label, req_claims, self.passages))
        verb = {"satisfied": "Respecte", "violated": "Ne respecte pas", "uncertain": "Respect incertain de"}[check.outcome]
        rule = "exigence documentée" if requirement is not None else "contrainte de la requête"
        return self.verifier.infer(f"{verb} {check.constraint.label}", premises, rule)

    def _value_case(
        self, draft: Draft, evidence: TrajectoryEvidence, relevance: Mapping[str, float]
    ) -> tuple[list[Premise], list[VerifiedClaim]]:
        assert draft.entity_key is not None and draft.attribute is not None and draft.group is not None
        group = draft.group
        attribute_passages = [c.passage_id for c in evidence.claims
                              if self.graph.canonical_key(c.subject_key) == draft.entity_key
                              and c.attribute == draft.attribute]
        premises = [
            Premise(label="Pertinence des passages (rang hybride)",
                    degree=max((relevance.get(p, 0.0) for p in attribute_passages), default=0.0), kind="relevance"),
            Premise(label="Appui des sources (OU bruité)", degree=group.support, kind="evidence"),
        ]
        quotes = " ".join(self.graph.claims[c].quote + " " + self.graph.claims[c].source.title for c in group.claim_ids)
        facet = self._facet_premise(quotes, evidence.interpretation)
        if facet is not None:
            premises.append(facet)
        resolution = self.graph.resolve(draft.entity_key, draft.attribute)
        if resolution.conflict and resolution.resolved and resolution.winner is not None \
                and resolution.winner.value_text != group.value_text:
            winner = resolution.winner
            premises.append(Premise(
                label=f"Contredit par {group_value(winner)} (appui {winner.support:.2f} contre {group.support:.2f})",
                degree=0.0, kind="consistency"))
            claim = VerifiedClaim(
                text=f"{attribute_label(draft.attribute)} : {group_value(group)}", status=Status.NON_VERIFIE,
                reason=f"Contredit par des sources plus fiables ({group_value(winner)}, appui {winner.support:.2f})",
                sources=tuple(self.graph.claims[c].source for c in group.claim_ids),
                claim_ids=group.claim_ids, confidence=0.0,
            )
            return premises, [claim]
        conflict_open = resolution.conflict and not resolution.resolved
        premises.append(Premise(label="Sources en désaccord" if conflict_open else "Aucune contradiction ouverte",
                                degree=UNKNOWN if conflict_open else 1.0, kind="consistency"))
        for constraint in self.plan.constraints:
            if constraint.attribute == draft.attribute and group.quantity is not None:
                verdict = satisfies(group.quantity, constraint.comparator, constraint.quantity)
                degree = UNKNOWN if verdict is None else (1.0 if verdict else 0.0)
                premises.append(Premise(label=constraint.label, degree=degree, kind="constraint"))
        verified = self.verifier.verify_attribute(self.graph, draft.entity_key, draft.attribute, self.passages)
        return premises, [verified] if verified is not None else []

    def _statement_case(
        self, draft: Draft, evidence: TrajectoryEvidence
    ) -> tuple[list[Premise], list[VerifiedClaim]]:
        assert draft.sentence is not None and draft.sentence_passage is not None
        passage = self.passages[draft.sentence_passage]
        sources = {passage.source.id: passage.source.reliability}
        sources.update({s.id: s.reliability for s in draft.corroborating})
        premises = [
            Premise(label="Admissibilité ARG de la phrase (reach, couverture)", degree=draft.arg_degree,
                    kind="relevance"),
            Premise(label="Appui des sources (OU bruité)", degree=round(noisy_or(sources.values()), 4),
                    kind="evidence"),
        ]
        facet = self._facet_premise(f"{draft.sentence} {passage.source.title}", evidence.interpretation)
        if facet is not None:
            premises.append(facet)
        disputed = self._sentence_disputed(draft.sentence, passage)
        premises.append(Premise(label="Une autre source donne une valeur différente" if disputed
                                else "Aucune contradiction ouverte",
                                degree=UNKNOWN if disputed else 1.0, kind="consistency"))
        claim = self.verifier.verify_statement(draft.sentence, draft.sentence, passage, draft.corroborating, disputed)
        return premises, [claim]

    def _sentence_disputed(self, sentence: str, passage: Passage) -> bool:
        for claim in self.graph.claims.values():
            if (
                claim.passage_id == passage.id and claim.quote == sentence
                and claim.attribute in self._open_conflicts.get(self.graph.canonical_key(claim.subject_key), [])
            ):
                return True
        return False


def support_degree(reliabilities: Iterable[float]) -> float:
    return round(1.0 - math.exp(-sum(reliabilities) / 2.0), 4)


def tap_score(
    admissibility: float,
    claims: Sequence[VerifiedClaim],
    reliabilities: Iterable[float],
    coherence: float,
    open_conflicts: int,
) -> TapScore:
    support = support_degree(reliabilities)
    faithfulness = sum(_STATUS_WEIGHT[c.status] for c in claims) / len(claims) if claims else UNKNOWN
    penalty = min(0.75, 0.25 * open_conflicts)
    quality = (coherence + support + faithfulness) / 3.0 * (1.0 - penalty)
    return TapScore(
        admissibility=round(admissibility, 4),
        coherence=round(coherence, 4),
        support=support,
        faithfulness=round(faithfulness, 4),
        contradiction_penalty=round(penalty, 4),
        quality=round(quality, 4),
        reliability=round(min(admissibility, quality), 4),
    )


def answer_for(candidate: Candidate, kind: str) -> str:
    if kind == "entity":
        return entity_answer(candidate.label, candidate.claims)
    if kind == "value":
        return value_answer(candidate.label, candidate.claims)
    return candidate.claims[0].text if candidate.claims else candidate.label


def select_possibilities(
    trajectories: Sequence[Trajectory],
    interpretations: Sequence[Interpretation],
    click_priors: Mapping[str, float],
    limit: int = 3,
    keep_ratio: float = 0.5,
) -> list[Possibility]:
    """2 ou 3 réponses : la meilleure de chaque trajectoire admissible, dédupliquée, triée par fiabilité."""
    by_id = {i.id: i for i in interpretations}
    pool: dict[str, tuple[Candidate, Trajectory, tuple[str, ...]]] = {}
    seconds: list[tuple[Candidate, Trajectory]] = []
    for trajectory in trajectories:
        if trajectory.best is None:
            continue
        kept = [c for c in trajectory.candidates if not c.rejected]
        alternatives = tuple(c.label for c in kept[1:4])
        best = trajectory.best
        current = pool.get(best.key)
        if current is None or best.score.reliability > current[0].score.reliability:
            pool[best.key] = (best, trajectory, alternatives)
        seconds.extend((c, trajectory) for c in kept[1:2])
    if len(pool) < 2:
        for candidate, trajectory in sorted(seconds, key=lambda x: x[0].score.reliability, reverse=True):
            if candidate.key not in pool:
                pool[candidate.key] = (candidate, trajectory, ())
            if len(pool) >= 2:
                break

    def facet_of(trajectory: Trajectory) -> str:
        interp = by_id.get(trajectory.interpretation_id)
        return interp.facet if interp else trajectory.interpretation_id

    ranked = sorted(
        pool.values(),
        key=lambda item: (
            round(item[0].score.reliability * 100),
            click_priors.get(facet_of(item[1]), 0.5),
            by_id[item[1].interpretation_id].plausibility if item[1].interpretation_id in by_id else 0.0,
        ),
        reverse=True,
    )
    if not ranked:
        return []
    top = ranked[0][0].score.reliability
    selected = [item for n, item in enumerate(ranked[:limit]) if n < 2 or item[0].score.reliability >= keep_ratio * top]
    out: list[Possibility] = []
    for rank, (candidate, trajectory, alternatives) in enumerate(selected, start=1):
        interp = by_id.get(trajectory.interpretation_id)
        kind = candidate.key.split(":", 1)[0]
        sources = {s.id: s for c in candidate.claims for s in c.sources}
        out.append(
            Possibility(
                rank=rank,
                interpretation_id=trajectory.interpretation_id,
                interpretation_label=interp.label if interp else trajectory.interpretation_id,
                title=candidate.label,
                answer=answer_for(candidate, kind),
                reliability=candidate.score.reliability,
                reliability_pct=round(candidate.score.reliability * 100),
                click_prior=round(click_priors.get(facet_of(trajectory), 0.5), 4),
                claims=candidate.claims,
                sources=tuple(sorted(sources.values(), key=lambda s: s.reliability, reverse=True)),
                alternatives=alternatives,
                synthesis="template",
            )
        )
    return out

