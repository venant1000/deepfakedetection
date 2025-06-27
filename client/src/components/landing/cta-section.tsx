import { useLocation } from "wouter";

export default function CTASection() {
  const [, navigate] = useLocation();
  
  return (
    <section className="py-20 px-6 relative gradient-bg">
      <div className="container mx-auto max-w-4xl text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-6">Ready to Detect <span className="neon-text">Deepfakes</span>?</h2>
        <p className="text-xl text-gray-200 mb-8 max-w-2xl mx-auto">Join thousands of users and organizations using DeepGuard AI to verify content authenticity and combat misinformation.</p>
        
      </div>
    </section>
  );
}
