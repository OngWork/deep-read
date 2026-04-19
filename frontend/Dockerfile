# Use Node.js for Build
FROM node:20-alpine

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

COPY . .

# expose port 5173 for Vite
EXPOSE 5173

# run Vite in dev mode and allow access from outside the container
CMD ["npm", "run", "dev", "--", "--host"]