// frontend/src/components/AnalyzeResult.tsx
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from "recharts";

export type RadarItem = { axis: string; value: number };
export type AnalyzeResponse = {
  // 기존 필드 (서버 구형 응답 호환용)
  rfpId?: string;
  proposalId?: string;
  rfpSummary?: string;
  matchRate?: number;
  ivi?: {
    overall: number;
    planning: number;
    feasibility: number;
    evidence: number;
    risk: number;
    clarity: number;
  };
  radar?: RadarItem[];
  feedback?: string[];
  decision?: "SUBMIT" | "HOLD" | "REWRITE";

  // 신규: 즉시평가 응답(instant)
  metrics?: Record<
    "CP" | "RI" | "FP" | "ETS" | "IO" | "RM",
    { metric?: string; score_10?: number; feedback?: string; raw?: string }
  >;
  metricsScores?: Record<"CP" | "RI" | "FP" | "ETS" | "IO" | "RM", number>;
  metricsTotal10?: number;
  similarity?: { similarity_percent?: number; feedback?: string; raw?: string } | null;
  guideReview?: { overall_score_10?: number; feedback?: string; raw?: string } | null;
};

function toRadarFromSix(scores?: AnalyzeResponse["metricsScores"]) {
  if (!scores) return [];
  return [
    { axis: "CP", value: scores.CP ?? 0 },
    { axis: "RI", value: scores.RI ?? 0 },
    { axis: "FP", value: scores.FP ?? 0 },
    { axis: "ETS", value: scores.ETS ?? 0 },
    { axis: "IO", value: scores.IO ?? 0 },
    { axis: "RM", value: scores.RM ?? 0 },
  ];
}

export default function AnalyzeResult({ data }: { data: AnalyzeResponse }) {
  // 신규 응답 기준 우선
  const six = data.metricsScores;
  const total10 = data.metricsTotal10 ?? 0;
  const radarData = six ? toRadarFromSix(six) : data.radar ?? [];
  const sim = data.similarity;
  const guide = data.guideReview;

  // 기존 배지 로직 (신규엔 없을 수 있으므로 안전 처리)
  const badge =
    data.decision === "SUBMIT"
      ? "bg-green-100 text-green-800"
      : data.decision === "HOLD"
      ? "bg-yellow-100 text-yellow-800"
      : data.decision === "REWRITE"
      ? "bg-red-100 text-red-800"
      : "bg-gray-100 text-gray-700";

  return (
    <div className="mt-10 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">분석 결과</h2>
        {data.decision && (
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${badge}`}>
            {data.decision === "SUBMIT" ? "제출 권장" : data.decision === "HOLD" ? "보류" : "재작성 필요"}
          </span>
        )}
      </div>

      {/* (옵션) RFP 요약 - 구형 응답 호환 */}
      {data.rfpSummary && (
        <div className="rounded-2xl border bg-white p-5">
          <div className="text-sm text-gray-500 mb-2">RFP 핵심 요약</div>
          <p className="leading-relaxed whitespace-pre-wrap">{data.rfpSummary}</p>
        </div>
      )}

      {/* 6개 영역 카드 + 총점 + 레이더 */}
      {six && (
        <>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <div className="rounded-2xl border bg-white p-5">
              <div className="text-sm text-gray-500">총점(6개 평균, 10점 만점)</div>
              <div className="mt-2 text-3xl font-semibold">{total10}</div>
              <ul className="mt-3 text-sm text-gray-600 space-y-1">
                <li>CP (목적 명확성): {six.CP}</li>
                <li>RI (관련성/임팩트): {six.RI}</li>
                <li>FP (실행가능성/계획): {six.FP}</li>
                <li>ETS (전문성/팀): {six.ETS}</li>
                <li>IO (혁신성/독창성): {six.IO}</li>
                <li>RM (리스크관리): {six.RM}</li>
              </ul>
            </div>


            <div className="rounded-2xl border bg-white p-5">
              <div className="text-sm text-gray-500 mb-2">Radar Chart (6개 영역, 10점 만점)</div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="axis" />
                    <Radar name="Score" dataKey="value" />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* 6개 영역 상세 피드백 */}
          <div className="rounded-2xl border bg-white p-5">
            <div className="text-sm text-gray-500 mb-2">6개 영역 평가 상세</div>
            <div className="space-y-4">
              {(["CP","RI","FP","ETS","IO","RM"] as const).map((k) => (
                <div key={k}>
                  <div className="font-medium">
                    {k} — 점수: {six[k] ?? "-"}
                  </div>
                  <div className="text-gray-800 whitespace-pre-wrap text-sm">
                    {data.metrics?.[k]?.feedback ?? data.metrics?.[k]?.raw ?? "(응답 없음)"}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* RFP-유사성 섹션 */}
      {sim && (
        <div className="rounded-2xl border bg-white p-5">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-500">RFP-제안서 유사성</div>
            <div className="text-2xl font-semibold">{sim.similarity_percent ?? "-"}%</div>
          </div>
          <div className="mt-3 text-gray-800 whitespace-pre-wrap text-sm">
            {sim.feedback ?? sim.raw ?? "(응답 없음)"}
          </div>
        </div>
      )}

      {/* 가이드 기반 종합평가 섹션 */}
      {guide && (
        <div className="rounded-2xl border bg-white p-5">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-500">가이드 기준 종합평가</div>
            <div className="text-2xl font-semibold">{guide.overall_score_10 ?? "-"}/10</div>
          </div>
          <div className="mt-3 text-gray-800 whitespace-pre-wrap text-sm">
            {guide.feedback ?? guide.raw ?? "(응답 없음)"}
          </div>
        </div>
      )}

      {/* (옵션) 구형 피드백 블록 유지 */}
      {data.feedback && data.feedback.length > 0 && (
        <div className="rounded-2xl border bg-white p-5">
          <div className="text-sm text-gray-500 mb-2">GPT 기반 피드백(간이)</div>
          <ul className="list-disc pl-5 space-y-1">
            {data.feedback.map((f, i) => (
              <li key={i} className="text-gray-800">{f}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
