import Navigation from "@/components/layout/navigation";
import HeroSection from "@/components/landing/hero-section";
import FeaturesSection from "@/components/landing/features-section";
import HowItWorks from "@/components/landing/how-it-works";
import TestimonialsSection from "@/components/landing/testimonials-section";
import CTASection from "@/components/landing/cta-section";
import ParticlesBackground from "@/components/particles-background";

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <ParticlesBackground />
      <Navigation />
      <HeroSection />
      <FeaturesSection />
      <HowItWorks />
      <TestimonialsSection />
      <CTASection />
    </div>
  );
}
