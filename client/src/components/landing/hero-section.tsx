import { Button } from "@/components/ui/button";
import { useLocation } from "wouter";

export default function HeroSection() {
  const [, navigate] = useLocation();
  
  return (
    <section className="pt-32 pb-20 px-6 relative grid-bg">
      <div className="container mx-auto max-w-6xl">
        <div className="flex flex-col lg:flex-row items-center gap-12">
          <div className="lg:w-1/2 space-y-8">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight">
              <span className="gradient-text">Detect Deepfakes</span> With Advanced AI Technology
            </h1>
            <p className="text-lg text-muted-foreground">
              Advanced AI technology to analyze videos and detect manipulation with precision and reliability.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <Button 
                className="py-6 px-8 rounded-lg bg-gradient-to-r from-primary to-secondary text-black font-semibold hover:opacity-90 button-glow transition-all text-base"
                onClick={() => navigate("/auth")}
              >
                Get Started
              </Button>
            </div>
            <div className="flex items-center gap-6 pt-4">
              <p className="text-sm text-muted-foreground">Powered by:</p>
              <div className="text-muted-foreground font-mono text-sm py-1 px-3 bg-background-light rounded-md">
                Google Gemini AI
              </div>
            </div>
          </div>
          
          <div className="lg:w-1/2 relative">
            {/* AI Analyzing interface */}
            <div className="glass rounded-2xl p-1 overflow-hidden neon-border shadow-xl">
              <div className="rounded-xl aspect-video bg-black/30 flex items-center justify-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="120"
                  height="120"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="rgba(0, 255, 136, 0.3)"
                  strokeWidth="0.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 8v8" />
                  <path d="m8.5 12 7 0" />
                </svg>
              </div>
              
              {/* Overlay Elements for Futuristic UI */}
              <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
                <div className="glass rounded-xl p-4 max-w-xs">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="h-3 w-3 rounded-full bg-primary animate-pulse"></div>
                    <span className="text-xs font-mono">DEEPFAKE ANALYSIS IN PROGRESS</span>
                  </div>
                  <div className="w-full bg-black/50 rounded-full h-1.5 mb-2">
                    <div className="bg-primary h-1.5 rounded-full w-3/4"></div>
                  </div>
                  <p className="text-xs text-muted-foreground font-mono">Analyzing frame patterns: 75% complete</p>
                </div>
              </div>
            </div>
            
            {/* Floating UI Elements */}
            <div className="absolute -top-5 -right-5 glass p-3 rounded-lg neon-border shadow-lg floating-card">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-[#ff3366]"></div>
                <span className="text-xs font-mono">Facial Inconsistency Detected</span>
              </div>
            </div>
            
            <div className="absolute -bottom-6 -left-6 glass p-3 rounded-lg neon-border-purple shadow-lg floating-card">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-[#ffbb00]"></div>
                <span className="text-xs font-mono">Audio-Visual Sync Mismatch</span>
              </div>
            </div>
          </div>
        </div>
        
        
      </div>
    </section>
  );
}
