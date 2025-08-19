export default function CenterTest() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-screen-lg mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="text-center mb-8 max-w-3xl mx-auto">
          <h1 className="text-2xl font-semibold">센터/2열 테스트</h1>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
          <div className="w-full max-w-md mx-auto h-40 rounded-2xl border-2 border-dashed" />
          <div className="w-full max-w-md mx-auto h-40 rounded-2xl border-2 border-dashed" />
        </div>
      </div>
    </div>
  );
}
