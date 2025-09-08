import React from "react";

function Navbar({ setPage }) {
  return (
    <nav className="flex justify-between items-center bg-white shadow-md p-4">
      <h1 className="text-xl font-bold text-purple-700">🫁 Pneumonia Analyzer</h1>
      <div>
        <button
          onClick={() => setPage("home")}
          className="px-3 py-1 mx-2 rounded bg-blue-500 text-white hover:bg-blue-600"
        >
          Home
        </button>
        <button
          onClick={() => setPage("analyzer")}
          className="px-3 py-1 mx-2 rounded bg-purple-500 text-white hover:bg-purple-600"
        >
          Analyzer
        </button>
      </div>
    </nav>
  );
}

export default Navbar;
