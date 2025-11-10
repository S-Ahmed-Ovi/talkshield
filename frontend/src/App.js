import React, { useState } from 'react';

function App() {
  const [text, setText] = useState('');
  const [spamResult, setSpamResult] = useState('');
  const [fraudResult, setFraudResult] = useState('');

  const handleSpamCheck = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/spam/predict/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      const data = await response.json();
      setSpamResult(data.prediction);
    } catch (err) {
      setSpamResult('Error connecting to backend');
      console.error(err);
    }
  };

  const handleFraudCheck = async () => {
    try {
      // Example: sending 30 dummy features for fraud prediction
      const response = await fetch('http://127.0.0.1:8000/api/fraud/predict/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: Array(30).fill(0) }) // replace with actual features
      });
      const data = await response.json();
      setFraudResult(data.prediction);
    } catch (err) {
      setFraudResult('Error connecting to backend');
      console.error(err);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Spam & Fraud Detection</h1>
      <textarea
        rows="4"
        cols="50"
        placeholder="Type message here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <br /><br />
      <button onClick={handleSpamCheck}>Check Spam</button>
      <button onClick={handleFraudCheck}>Check Fraud</button>
      <h2>Spam Result: {spamResult}</h2>
      <h2>Fraud Result: {fraudResult}</h2>
    </div>
  );
}

export default App;
