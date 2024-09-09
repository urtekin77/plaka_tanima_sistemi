import React, { useState } from 'react';
import './App.css';


function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult('');
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError('Lütfen bir resim seçin');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);


    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.result) {
        setResult(data.result);
      } else if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError('Sunucuya bağlanırken bir hata oluştu.');
    }
  };

  return (
    <div className="App">
      <h1>Plaka Tanıma Sistemi</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Gönder</button>
      </form>
      {result && <h3>Tespit Edilen Plaka: {result}</h3>}
      {error && <h3 style={{ color: 'red' }}>Hata:{error}</h3>}
    </div>
  );
}

export default App;
