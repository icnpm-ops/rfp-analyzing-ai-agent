export async function pingBackend() {
  const res = await fetch("http://localhost:8000/");
  return res.json();
}
