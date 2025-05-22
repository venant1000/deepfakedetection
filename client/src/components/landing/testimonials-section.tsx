import { Shield, Clock, Target, Users } from "lucide-react";

export default function TestimonialsSection() {
  const stats = [
    {
      icon: Shield,
      number: "99.8%",
      label: "Detection Accuracy",
      description: "Advanced AI algorithms ensure precise deepfake identification",
      delay: "0s"
    },
    {
      icon: Clock,
      number: "< 30s",
      label: "Analysis Time",
      description: "Lightning-fast processing for real-time content verification",
      delay: "0.3s"
    },
    {
      icon: Target,
      number: "50+",
      label: "Manipulation Types",
      description: "Comprehensive detection of various deepfake techniques",
      delay: "0.6s"
    }
  ];

  return (
    <section className="py-20 px-6 bg-black/30">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            <span className="gradient-text">Trusted Performance</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Industry-leading AI technology delivering reliable results for content verification
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {stats.map((stat, index) => (
            <div 
              key={index} 
              className="glass rounded-xl p-8 floating-card text-center hover:scale-105 transition-transform duration-300" 
              style={{animationDelay: stat.delay}}
            >
              <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center">
                <stat.icon className="w-8 h-8 text-white" />
              </div>
              <div className="text-4xl font-bold gradient-text mb-3">
                {stat.number}
              </div>
              <h3 className="text-xl font-semibold mb-3">
                {stat.label}
              </h3>
              <p className="text-muted-foreground text-sm">
                {stat.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
