import React from "react";
import pneumoniaImg from "../assets/jZqpV51.png"; // place image in /src/assets/

function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-white">
      {/* Header */}
      <header className="text-center py-12">
        <h1 className="text-4xl font-extrabold mb-4">🩻 Pneumonia Detection System</h1>
        <p className="text-lg max-w-2xl mx-auto">
          Upload your chest X-ray and let our AI-powered system help identify 
          whether it’s <span className="font-semibold">Normal</span>, 
          <span className="font-semibold"> Bacterial Pneumonia</span>, or 
          <span className="font-semibold"> Viral Pneumonia</span>.
        </p>
      </header>

      {/* Reference Image Section */}
      <section className="max-w-5xl mx-auto bg-white text-black rounded-lg shadow-xl p-8">
        <h2 className="text-2xl font-bold text-center mb-6">
          📊 Reference X-ray Images
        </h2>
        <img
          src={pneumoniaImg}
          alt="Normal vs Bacterial vs Viral Pneumonia"
          className="rounded-lg shadow-md mx-auto"
        />
        <p className="mt-6 text-center text-gray-700">
          These sample X-rays illustrate the visual differences between{" "}
          <span className="font-semibold">Normal lungs</span>,{" "}
          <span className="font-semibold">Bacterial Pneumonia</span>, and{" "}
          <span className="font-semibold">Viral Pneumonia</span>.
        </p>
      </section>

      {/* Info Cards */}
      <section className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8 px-8 max-w-6xl mx-auto">
        <div className="bg-white text-black rounded-lg p-6 shadow-lg hover:scale-105 transform transition">
          <h3 className="text-xl font-semibold mb-3">🔹 Normal</h3>
          <p>
            Healthy lungs with clear airways. No abnormal opacities or
            infections detected. AI should classify this as safe.
          </p>
        </div>
        <div className="bg-white text-black rounded-lg p-6 shadow-lg hover:scale-105 transform transition">
          <h3 className="text-xl font-semibold mb-3">🦠 Bacterial Pneumonia</h3>
          <p>
            Caused by bacteria such as <em>Streptococcus pneumoniae</em>. 
            X-rays show localized, dense white patches indicating infection.
          </p>
        </div>
        <div className="bg-white text-black rounded-lg p-6 shadow-lg hover:scale-105 transform transition">
          <h3 className="text-xl font-semibold mb-3">🧬 Viral Pneumonia</h3>
          <p>
            Caused by viruses such as Influenza or COVID-19. X-rays show
            widespread hazy opacities, more diffuse than bacterial pneumonia.
          </p>
        </div>
      </section>

      {/* Call to Action */}
      <section className="mt-16 text-center">
        <h2 className="text-3xl font-bold mb-4">🩺 Ready to Analyze?</h2>
        <p className="mb-6">
          Click below to upload your chest X-ray and get instant AI insights.
        </p>
        <a
          href="/analyzer"
          className="px-6 py-3 bg-yellow-400 text-black rounded-full font-semibold text-lg shadow-lg hover:bg-yellow-500 transition"
        >
          Upload & Analyze
        </a>
      </section>

      {/* Footer */}
      <footer className="mt-16 py-6 text-center text-sm text-gray-200">
        © 2025 Pneumonia Analyzer | Built with ❤️ by Henry
      </footer>
    </div>
  );
}

export default HomePage;
