import React from "react";

function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-white">
      <header className="text-center py-12">
        <h1 className="text-4xl font-extrabold mb-4">🩻 Pneumonia Detection System</h1>
        <p className="text-lg max-w-2xl mx-auto">
          Upload your chest X-ray and let our AI-powered system help identify 
          whether it’s <span className="font-semibold">Normal</span> or{" "}
          <span className="font-semibold">Pneumonia</span>.
        </p>
      </header>

      <section className="max-w-5xl mx-auto bg-white text-black rounded-lg shadow-xl p-8">
        <h2 className="text-2xl font-bold text-center mb-6">
          📊 Reference X-ray Images
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="relative">
            <img
            style={{width: '100%', height: '450px'}}
              src="https://prod-images-static.radiopaedia.org/images/220869/76052f7902246ff862f52f5d3cd9cd_big_gallery.jpg"
              alt="Normal Lung"
              className="rounded-lg shadow-md mx-auto"
            />
            <span className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-green-600 text-white px-3 py-1 rounded-md font-semibold">
              Normal
            </span>
          </div>
          <div className="relative">
            <img
              src="https://prod-images-static.radiopaedia.org/images/1371188/0a1f5edc85aa58d5780928cb39b08659c1fc4d6d7c7dce2f8db1d63c7c737234_big_gallery.jpeg"
              alt="Pneumonia Lung"
              className="rounded-lg shadow-md mx-auto"
            />
            <span className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-red-600 text-white px-3 py-1 rounded-md font-semibold">
              Pneumonia
            </span>
          </div>
        </div>

        <p className="mt-6 text-center text-gray-700">
          These examples illustrate the difference between{" "}
          <span className="font-semibold">Healthy lungs</span> and{" "}
          <span className="font-semibold">Pneumonia-affected lungs</span>.
        </p>
      </section>

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
      <footer className="mt-16 py-6 text-center text-sm text-gray-200">
         Pneumonia Analyzer | Built with ❤️ by HackSparrow
      </footer>
    </div>
  );
}

export default HomePage;
