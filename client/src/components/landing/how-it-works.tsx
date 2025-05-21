export default function HowItWorks() {
  const steps = [
    {
      number: 1,
      title: "Upload Your Video",
      description: "Simply drag and drop or select the video file you want to analyze."
    },
    {
      number: 2,
      title: "AI Analysis",
      description: "Our system uses Gemini AI to examine facial movements, audio-visual sync, and digital artifacts frame by frame."
    },
    {
      number: 3,
      title: "Review Results",
      description: "Get a detailed report showing confidence scores, problematic segments, and specific inconsistencies."
    },
    {
      number: 4,
      title: "Export and Share",
      description: "Download a comprehensive PDF report or share results securely with others."
    }
  ];

  return (
    <section id="how-it-works" className="py-20 px-6 relative">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">How DeepGuard <span className="gradient-text">Works</span></h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">Our technology combines Google's Gemini AI with proprietary algorithms to detect even the most sophisticated deepfakes.</p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="order-2 lg:order-1">
            <div className="glass rounded-xl p-6 space-y-8">
              {steps.map((step) => (
                <div key={step.number} className="flex items-start gap-6">
                  <div className="h-10 w-10 rounded-full bg-primary flex items-center justify-center flex-shrink-0 text-black font-bold">
                    {step.number}
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                    <p className="text-muted-foreground">{step.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="order-1 lg:order-2">
            {/* Digital security concept image */}
            <div className="glass rounded-2xl p-1 neon-border">
              <div className="rounded-xl aspect-video bg-black/30 relative overflow-hidden">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="100%"
                  height="100%"
                  viewBox="0 0 24 24"
                  fill="none"
                  className="absolute inset-0 text-primary/10"
                >
                  <path 
                    fill="currentColor"
                    d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"
                  />
                  <circle cx="12" cy="12" r="4" fill="rgba(0,255,136,0.3)" />
                  <path
                    stroke="rgba(0,255,136,0.5)"
                    strokeWidth="0.5"
                    d="M3 7h18M3 12h18M3 17h18"
                  />
                </svg>
                
                {/* Overlay Elements */}
                <div className="absolute top-1/4 left-1/4 glass p-3 rounded-lg shadow-lg animate-pulse">
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-primary"></div>
                    <span className="text-xs font-mono">Analyzing Facial Patterns</span>
                  </div>
                </div>
                
                <div className="absolute bottom-1/4 right-1/4 glass p-3 rounded-lg shadow-lg animate-pulse" style={{animationDelay: "1s"}}>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-primary"></div>
                    <span className="text-xs font-mono">Audio Consistency Check</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
