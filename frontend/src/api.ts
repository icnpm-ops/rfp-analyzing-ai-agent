const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://web-production-2a58.up.railway.app';

export async function pingBackend() {
  const res = await fetch(`${API_BASE_URL}/`);
  return res.json();
}

// 다른 API 함수들도 API_BASE_URL을 사용하도록 수정
export async function uploadFile(file: File, title: string, docType: string) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('title', title);
  formData.append('docType', docType);
  
  const res = await fetch(`${API_BASE_URL}upload`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
}