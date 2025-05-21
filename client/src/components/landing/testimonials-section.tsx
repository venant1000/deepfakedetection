export default function TestimonialsSection() {
  const testimonials = [
    {
      content: "DeepGuard saved our news organization from publishing a deepfake video. The analysis was quick and the explanations were clear. Invaluable tool in the fight against misinformation.",
      author: "Jessica Lee",
      role: "News Editor, TechNews",
      delay: "0s"
    },
    {
      content: "As a content moderator, DeepGuard AI has become essential to my workflow. The timeline feature pinpoints exactly where manipulations occur, saving me hours of manual inspection.",
      author: "Marcus Johnson",
      role: "Content Moderator, SocialStream",
      delay: "0.3s"
    },
    {
      content: "The chatbot feature is surprisingly helpful. When I'm unsure about a result, I can ask for more information and get detailed explanations. The UI is beautiful and intuitive.",
      author: "Aisha Patel",
      role: "Digital Investigator",
      delay: "0.6s"
    }
  ];

  return (
    <section className="py-20 px-6 bg-black/30">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">What Our <span className="gradient-text">Users Say</span></h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">Join thousands of satisfied users already protecting themselves against deepfakes.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {testimonials.map((testimonial, index) => (
            <div 
              key={index} 
              className="glass rounded-xl p-6 floating-card" 
              style={{animationDelay: testimonial.delay}}
            >
              <div className="flex items-center gap-2 mb-4">
                {[...Array(5)].map((_, i) => (
                  <svg 
                    key={i} 
                    xmlns="http://www.w3.org/2000/svg" 
                    width="16" 
                    height="16" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round" 
                    className="text-primary"
                  >
                    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                  </svg>
                ))}
              </div>
              <p className="text-muted-foreground mb-6">{testimonial.content}</p>
              <div className="flex items-center gap-4">
                <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                </div>
                <div>
                  <p className="font-semibold">{testimonial.author}</p>
                  <p className="text-sm text-muted-foreground">{testimonial.role}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
