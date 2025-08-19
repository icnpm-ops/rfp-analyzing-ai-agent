// frontend/src/components/FileUpload.tsx
import { useRef, useState } from "react";
import type { DragEvent, ChangeEvent } from "react";
import axios, { type AxiosProgressEvent } from "axios";
import AnalyzeResult, { type AnalyzeResponse } from "./AnalyzeResult";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const ACCEPTED_EXT = [".pdf", ".docx"];
const MAX_BYTES = 20 * 1024 * 1024;

type ZoneKind = "RFP" | "PROPOSAL";

type DropZoneProps = {
  kind: ZoneKind;
  file: File | null;
  onSelect: (file: File | null) => void;
  error?: string;
};

function humanSize(bytes: number) {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}
function stripExt(name: string) {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(0, i) : name;
}


function DropZone({ kind, file, onSelect, error }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const label = kind === "RFP" ? "RFP 업로드" : "제안서(초안) 업로드";
  const desc = kind === "RFP" ? "요구서(RFP) 문서를 드래그하거나 클릭해서 선택" : "제안서(초안) 문서를 드래그하거나 클릭해서 선택";

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) onSelect(f);
  };
  const onChange = (e: ChangeEvent<HTMLInputElement>) => onSelect(e.target.files?.[0] ?? null);
  const onOpen = () => inputRef.current?.click();

  return (
    <div className="w-full">
      <div className="mb-2 text-sm font-medium text-gray-700">{label}</div>
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={onOpen}
        role="button"
        aria-label={label}
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " " ? onOpen() : null)}
        className="flex h-40 items-center justify-center rounded-2xl border-2 border-dashed border-gray-300 bg-white hover:border-blue-500"
      >
        {!file ? (
          <div className="text-center px-4">
            <div className="text-gray-600">{desc}</div>
            <div className="mt-1 text-xs text-gray-400">허용: .pdf, .docx (최대 20MB)</div>
          </div>
        ) : (
          <div className="text-center px-4">
            <div className="text-gray-800 font-medium">{file.name}</div>
            <div className="text-xs text-gray-500 mt-1">{humanSize(file.size)}</div>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onSelect(null); }}
              className="mt-3 rounded-lg border px-3 py-1.5 text-sm hover:bg-gray-50"
            >
              파일 제거
            </button>
          </div>
        )}
        <input ref={inputRef} type="file" accept={ACCEPTED_EXT.join(",")} className="hidden" onChange={onChange} />
      </div>
      {error && <div className="mt-1 text-xs text-red-600">{error}</div>}
    </div>
  );
}

export default function FileUpload() {
  const [rfp, setRfp] = useState<File | null>(null);
  const [proposal, setProposal] = useState<File | null>(null);
  const [errRfp, setErrRfp] = useState("");
  const [errProp, setErrProp] = useState("");

  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const validate = (file: File | null) => {
    if (!file) return "파일이 선택되지 않았어요.";
    const lower = file.name.toLowerCase();
    if (!ACCEPTED_EXT.some((ext) => lower.endsWith(ext))) return "허용되지 않는 형식입니다. (.pdf, .docx)";
    if (file.size > MAX_BYTES) return "파일 용량이 20MB를 초과합니다.";
    return "";
  };

  const onSelectRfp = (f: File | null) => { setErrRfp(""); if (!f) return setRfp(null); const v = validate(f); v ? (setErrRfp(v), setRfp(null)) : setRfp(f); };
  const onSelectProp = (f: File | null) => { setErrProp(""); if (!f) return setProposal(null); const v = validate(f); v ? (setErrProp(v), setProposal(null)) : setProposal(f); };

  const canProcess = !!rfp && !!proposal && !isProcessing;

// 타입: 백엔드 /upload 응답과 일치
type UploadResponse = {
  ok: boolean;
  ready?: boolean;            // 변환 완료 신호
  docID: string;              // 업로드된 문서 ID
  storeAs: string;            // 원본 저장 경로
  txtPath?: string;           // ★ 생성된 .extracted.txt 경로
  message?: string;
};

// 수정된 uploadOne: docID만이 아니라 전체 응답을 반환
const uploadOne = async (
  file: File,
  docType: "RFP" | "Proposal",
  progressOffset: number
): Promise<UploadResponse> => {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("title", stripExt(file.name));
  fd.append("docType", docType);

  const res = await axios.post(`${API_BASE}/upload`, fd, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (evt: AxiosProgressEvent) => {
      const total = evt.total ?? 0;
      if (!total) return;
      const pct = Math.round(((evt.loaded ?? 0) * 100) / total);
      // 업로드 구간 진행률: 0~40 / 40~80
      setProgress(progressOffset + Math.floor(pct * 0.4));
    },
  });

  return res.data as UploadResponse;
};


const onProcess = async () => {
  if (!rfp || !proposal) return;
  setResult(null);
  setMessage("");
  setIsProcessing(true);
  setProgress(0);

  try {
    // 1) 업로드 2회 (txtPath 포함 응답)
    const rfpUp = await uploadOne(rfp, "RFP", 0);
    const propUp = await uploadOne(proposal, "Proposal", 40);

    // 2) 안전 확인 (동기 변환이므로 보통 true지만 가드 추가)
    if (!rfpUp.txtPath || !propUp.txtPath) {
      setMessage("텍스트 변환이 아직 완료되지 않았어요. 잠시 후 다시 시도해주세요.");
      return;
    }

    // 3) 즉시 평가 호출
    setProgress(80);
    setMessage("분석 중…");
    const evalRes = await axios.post(`${API_BASE}/evaluate/instant`, {
      proposalPath: propUp.txtPath,
      rfpPath: rfpUp.txtPath,
      guidePath: "guide/guide_reference.txt",
    });

    // 4) 결과 표시
    setResult(evalRes.data as AnalyzeResponse);
    setMessage("분석이 완료되었습니다.");
    setProgress(100);
  } catch (e: any) {
    console.error(e);
    setMessage(e?.response?.data?.detail ?? "처리 중 오류가 발생했습니다.");
  } finally {
    setIsProcessing(false);
  }
};


  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto w-full max-w-4xl px-4 py-10">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-semibold">RFP · 제안서 자동 분석</h1>
          <p className="mt-2 text-sm text-gray-600">
            왼쪽은 <span className="font-medium">RFP</span>, 오른쪽은 <span className="font-medium">제안서(초안)</span>을 업로드하세요.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <DropZone kind="RFP" file={rfp} onSelect={onSelectRfp} error={errRfp} />
          <DropZone kind="PROPOSAL" file={proposal} onSelect={onSelectProp} error={errProp} />
        </div>

        <div className="mt-8">
          {isProcessing && (
            <div className="mb-6 flex flex-col items-center">
              <svg className="h-8 w-8 animate-spin" viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" className="opacity-25" />
                <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="4" fill="none" className="opacity-75" />
              </svg>
              <div className="mt-3 w-full max-w-xl">
                <div className="text-sm text-gray-700 text-center">진행 중… {progress}%</div>
                <div className="mt-2 h-2 w-full rounded-full bg-gray-200">
                  <div className="h-2 rounded-full bg-blue-600 transition-all" style={{ width: `${progress}%` }} />
                </div>
              </div>
            </div>
          )}

          {message && <div className="mb-4 text-center text-sm text-gray-700">{message}</div>}

          <div className="flex items-center justify-center">
            <button
              onClick={onProcess}
              disabled={!canProcess}
              className={`rounded-xl px-6 py-3 text-white transition ${canProcess ? "bg-blue-600 hover:bg-blue-700" : "bg-gray-400 cursor-not-allowed"}`}
            >
              업로드 & 분석 시작
            </button>
          </div>

          {result && <AnalyzeResult data={result} />}
        </div>
      </div>
    </div>
  );
}
