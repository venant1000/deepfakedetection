import { Button } from "@/components/ui/button";
import { useLocation } from "wouter";

export default function CTASection() {
  const [, navigate] = useLocation();
  
  return (
    <section className="py-20 px-6 relative gradient-bg">
      <div className="container mx-auto max-w-4xl text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Detect <span className="neon-text">Deepfakes</span>?</h2>
        <p className="text-xl text-gray-200 mb-8 max-w-2xl mx-auto">Join thousands of users and organizations using DeepGuard AI to verify content authenticity and combat misinformation.</p>
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <Button 
            className="py-6 px-8 rounded-lg bg-primary text-black font-semibold hover:opacity-90 button-glow transition-all text-base"
            onClick={() => navigate("/auth")}
          >
            Get Started Free
          </Button>
          <Button 
            variant="outline" 
            className="py-6 px-8 rounded-lg border border-white text-white font-semibold hover:bg-white hover:text-background-dark transition-all text-base"
            onClick={() => navigate("/auth")}
          >
            Request Demo
          </Button>
        </div>
      </div>
    </section>
  );
}
