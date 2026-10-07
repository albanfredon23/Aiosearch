# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/build_ca; fi \
    && npm ci --no-audit --no-fund
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
COPY public ./public
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.31-alpine AS runtime
ENV NGINX_ENVSUBST_FILTER=^AIOTECH_
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --chmod=755 nginx/15-aiotech-web-key.envsh /docker-entrypoint.d/15-aiotech-web-key.envsh
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:8080/healthz"]
