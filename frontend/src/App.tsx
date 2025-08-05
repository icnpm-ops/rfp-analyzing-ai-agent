// src/App.tsx

import { useEffect } from 'react';
import { pingBackend } from './api'; // 경로는 실제 위치에 따라 조정

function App() {
  useEffect(() => {
    pingBackend().then(console.log);
  }, []);

  return <div>Hello from React</div>;
}

export default App;
