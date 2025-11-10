import axios from "axios";

const axiosClient = axios.create({
  baseURL: "http://127.0.0.1:8000", // Django base URL
});

export default axiosClient;
