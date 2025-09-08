import React, { useState } from "react";
import Home from "./pages/Home";
import Analyzer from "./pages/Analyzer";
import Navbar from "./components/Navbar";

function App() {
  const [page, setPage] = useState("home");

  return (
    <div>
      <Navbar setPage={setPage} />
      {page === "home" && <Home />}
      {page === "analyzer" && <Analyzer />}
    </div>
  );
}

export default App;
  