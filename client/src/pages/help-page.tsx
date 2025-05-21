import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { useToast } from "@/hooks/use-toast";

export default function HelpPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [contactForm, setContactForm] = useState({
    subject: "",
    message: "",
  });
  
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Search initiated",
      description: `Searching help articles for "${searchQuery}"`,
    });
  };
  
  const handleContactSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    toast({
      title: "Support request sent",
      description: "We've received your message and will respond shortly.",
    });
    
    setContactForm({
      subject: "",
      message: "",
    });
  };

  const faqItems = [
    {
      question: "What is DeepGuard AI?",
      answer: "DeepGuard AI is a cutting-edge platform that uses advanced artificial intelligence to detect deepfake videos. Our technology analyzes various aspects of video content to identify signs of manipulation, helping users verify the authenticity of media they encounter online."
    },
    {
      question: "How accurate is the deepfake detection?",
      answer: "DeepGuard AI uses Google's Gemini technology to achieve high accuracy rates in deepfake detection. While no system is 100% perfect, our platform typically achieves 85-95% accuracy depending on video quality and the sophistication of the manipulation. We provide confidence scores with each analysis to help you interpret results."
    },
    {
      question: "What types of deepfakes can DeepGuard AI detect?",
      answer: "Our platform can detect various types of deepfakes, including facial swaps, lip-sync manipulations, voice synthesis, and full body replacements. The system is constantly improving through machine learning to catch even the most sophisticated deepfakes."
    },
    {
      question: "What file formats are supported for upload?",
      answer: "DeepGuard AI supports most common video formats, including MP4, MOV, AVI, and WEBM. For optimal results, we recommend uploading videos with a resolution of at least 720p."
    },
    {
      question: "Are my uploaded videos kept private?",
      answer: "Yes, privacy is a core value of DeepGuard AI. Your videos are processed securely, and depending on your privacy settings, can be automatically deleted after analysis. We do not share your content with third parties unless explicitly authorized."
    },
    {
      question: "Is there a limit to how many videos I can analyze?",
      answer: "Standard accounts can analyze up to 10 videos per day. Premium accounts have higher or unlimited quotas depending on the subscription tier. Each video can be up to 100MB in size and 5 minutes in duration."
    },
    {
      question: "How do I interpret the analysis results?",
      answer: "Analysis results include an overall authenticity verdict, a confidence score, and specific findings detailing potential manipulation areas. The timeline feature highlights specific segments of the video where issues were detected, allowing for precise examination."
    },
  ];

  // Filter FAQs based on search query
  const filteredFaqs = searchQuery 
    ? faqItems.filter(item => 
        item.question.toLowerCase().includes(searchQuery.toLowerCase()) || 
        item.answer.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : faqItems;

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isAdmin={user?.role === "admin"} />
      
      <div className="flex-1 ml-20 md:ml-64 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-8">Help & Support</h1>
          
          <div className="glass rounded-xl p-8 mb-8">
            <h2 className="text-xl font-semibold mb-4">Search Help Articles</h2>
            <form onSubmit={handleSearch} className="flex gap-2">
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for help topics..."
                className="flex-1"
              />
              <Button type="submit">Search</Button>
            </form>
          </div>
          
          <div className="glass rounded-xl overflow-hidden mb-8">
            <div className="p-6 border-b border-muted">
              <h2 className="text-xl font-semibold">Frequently Asked Questions</h2>
            </div>
            
            <div className="p-6">
              {filteredFaqs.length > 0 ? (
                <Accordion type="single" collapsible className="w-full">
                  {filteredFaqs.map((item, index) => (
                    <AccordionItem key={index} value={`item-${index}`}>
                      <AccordionTrigger className="text-left">
                        {item.question}
                      </AccordionTrigger>
                      <AccordionContent>
                        <p className="text-muted-foreground">{item.answer}</p>
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              ) : (
                <div className="text-center py-8">
                  <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                  </div>
                  <h3 className="text-lg font-medium mb-2">No results found</h3>
                  <p className="text-muted-foreground">
                    We couldn't find any articles matching your search.
                  </p>
                </div>
              )}
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8 mb-8">
            <div className="glass rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2 text-primary"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                Quick Start Guide
              </h3>
              <ul className="space-y-2 mb-4">
                <li className="flex items-start gap-2">
                  <span className="bg-primary/10 text-primary h-5 w-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">1</span>
                  <span>Upload a video file on the Upload page</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-primary/10 text-primary h-5 w-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">2</span>
                  <span>Wait for the AI analysis to complete</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-primary/10 text-primary h-5 w-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">3</span>
                  <span>Review the detailed analysis results</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-primary/10 text-primary h-5 w-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">4</span>
                  <span>Save or export the analysis report</span>
                </li>
              </ul>
              <Button variant="outline" className="w-full">View Full Guide</Button>
            </div>
            
            <div className="glass rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2 text-primary"><path d="M17 18a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v9Z"/><path d="m17 2-4 4-4-4"/></svg>
                Video Tutorials
              </h3>
              <ul className="space-y-3 mb-4">
                <li className="flex items-center gap-3">
                  <div className="h-10 w-16 bg-muted rounded flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="6 3 18 12 6 21 6 3"/></svg>
                  </div>
                  <span>Getting Started with DeepGuard AI</span>
                </li>
                <li className="flex items-center gap-3">
                  <div className="h-10 w-16 bg-muted rounded flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="6 3 18 12 6 21 6 3"/></svg>
                  </div>
                  <span>Understanding Analysis Results</span>
                </li>
                <li className="flex items-center gap-3">
                  <div className="h-10 w-16 bg-muted rounded flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="6 3 18 12 6 21 6 3"/></svg>
                  </div>
                  <span>Advanced Detection Features</span>
                </li>
              </ul>
              <Button variant="outline" className="w-full">View All Tutorials</Button>
            </div>
          </div>
          
          <div className="glass rounded-xl overflow-hidden">
            <div className="p-6 border-b border-muted">
              <h2 className="text-xl font-semibold">Contact Support</h2>
            </div>
            
            <div className="p-6">
              <form onSubmit={handleContactSubmit} className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="subject" className="block font-medium">
                    Subject
                  </label>
                  <Input
                    id="subject"
                    value={contactForm.subject}
                    onChange={(e) => setContactForm({...contactForm, subject: e.target.value})}
                    placeholder="What can we help you with?"
                    required
                  />
                </div>
                
                <div className="space-y-2">
                  <label htmlFor="message" className="block font-medium">
                    Message
                  </label>
                  <textarea
                    id="message"
                    value={contactForm.message}
                    onChange={(e) => setContactForm({...contactForm, message: e.target.value})}
                    placeholder="Please describe your issue in detail..."
                    className="w-full min-h-[150px] p-3 rounded-md border border-input bg-transparent"
                    required
                  ></textarea>
                </div>
                
                <Button type="submit" className="w-full md:w-auto">
                  Submit Request
                </Button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}