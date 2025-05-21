export default function FeaturesSection() {
  const features = [
    {
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>,
      title: "Deepfake Detection",
      description: "Identify manipulated videos with state-of-the-art algorithms that detect visual inconsistencies."
    },
    {
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>,
      title: "Explainable Results",
      description: "Understand why content was flagged with visual breakdowns and detailed AI explanations."
    },
    {
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
      title: "AI Chatbot Support",
      description: "Get instant answers about deepfakes from our Gemini-powered educational chatbot."
    },
    {
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>,
      title: "Analytical Insights",
      description: "Access comprehensive reports and track deepfake detection patterns over time."
    }
  ];

  return (
    <section id="features" className="py-20 px-6 bg-black/30">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Advanced <span className="gradient-text">AI-Powered Features</span></h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">DeepGuard combines cutting-edge AI with user-friendly tools to protect against sophisticated deepfakes.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <div key={index} className="glass rounded-xl p-6 hover:translate-y-[-5px] transition-all">
              <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-primary to-[#0088ff] flex items-center justify-center mb-4">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
              <p className="text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
