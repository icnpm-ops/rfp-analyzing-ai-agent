/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  // 필요시 다른 환경변수 추가
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}