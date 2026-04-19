# ใช้ Node.js สำหรับ Build
FROM node:20-alpine

WORKDIR /app

# Copy package files และติดตั้ง dependencies
COPY package*.json ./
RUN npm install

# Copy โค้ดทั้งหมด
COPY . .

# เปิดพอร์ต 5173 สำหรับ Vite
EXPOSE 5173

# รัน Vite ในโหมด dev และอนุญาตให้เข้าถึงจากภายนอก container
CMD ["npm", "run", "dev", "--", "--host"]