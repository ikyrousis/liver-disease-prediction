FROM node:22-slim
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY public/ public/
COPY src/ src/

EXPOSE 3000
CMD ["npm", "start"]
