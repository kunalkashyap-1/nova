import axios from 'axios';
const api = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_API_URL,
  timeout: 30000,
});

export default api;