import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

export default function HelpPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  
  // FAQ categories and questions
  const faqCategories = [
    {
      id: "general",
      name: "General Questions",
      questions: [
        {
          id: "what-is",
          question: "What is DeepGuard AI?",
          answer: "DeepGuard AI is a cutting-edge platform that uses advanced artificial intelligence to detect deepfake videos and manipulated media. Our technology leverages Google's Gemini API to analyze videos and identify signs of manipulation with high accuracy."
        },
        {
          id: "how-it-works",
          question: "How does deepfake detection work?",
          answer: "Our deepfake detection system works by analyzing multiple aspects of a video, including facial inconsistencies, unnatural movements, lighting discrepancies, and audio-visual synchronization issues. The AI models have been trained on thousands of examples of both authentic and manipulated media to accurately identify signs of manipulation."
        },
        {
          id: "accuracy",
          question: "How accurate is the detection?",
          answer: "Our system currently achieves over 95% accuracy on most common types of deepfakes. However, as deepfake technology evolves, we continuously update our models to maintain high detection accuracy. We provide confidence scores with each analysis to indicate the reliability of our detection."
        }
      ]
    },
    {
      id: "account",
      name: "Account & Billing",
      questions: [
        {
          id: "free-tier",
          question: "What's included in the free tier?",
          answer: "The free tier includes up to 5 video analyses per month, limited to videos under 2 minutes in length. You also get access to basic reports and educational resources."
        },
        {
          id: "subscription",
          question: "How do I upgrade my subscription?",
          answer: "You can upgrade your subscription by visiting your Account Settings page and selecting the 'Upgrade' option. We offer several plans to suit different needs, including monthly and annual billing options."
        },
        {
          id: "cancel",
          question: "How do I cancel my subscription?",
          answer: "You can cancel your subscription at any time from your Account Settings page. Your premium features will remain available until the end of your current billing period."
        }
      ]
    },
    {
      id: "usage",
      name: "Using the Platform",
      questions: [
        {
          id: "upload-video",
          question: "How do I upload a video for analysis?",
          answer: "To upload a video, go to the Dashboard and click the 'Upload Video' button or drag and drop your video file into the designated area. Our system accepts most common video formats including MP4, MOV, and AVI files up to 100MB in size."
        },
        {
          id: "analysis-time",
          question: "How long does analysis take?",
          answer: "Analysis time depends on the length and complexity of the video, but typically ranges from 30 seconds to 5 minutes. You'll receive a notification when your analysis is complete."
        },
        {
          id: "understand-results",
          question: "How do I interpret the analysis results?",
          answer: "Our analysis results page provides a comprehensive breakdown of the detection findings. The main verdict indicates whether the video is likely authentic or manipulated, accompanied by a confidence score. You'll also see detailed information about specific issues detected, marked regions in the video, and technical explanations."
        }
      ]
    },
    {
      id: "technical",
      name: "Technical Support",
      questions: [
        {
          id: "supported-formats",
          question: "What video formats are supported?",
          answer: "We support most common video formats including MP4, MOV, AVI, WMV, and MKV. For optimal analysis, we recommend using MP4 files with H.264 encoding."
        },
        {
          id: "size-limits",
          question: "Is there a file size limit?",
          answer: "Yes, the maximum file size is 100MB for free tier users and 500MB for premium users. If your video exceeds these limits, consider compressing it or trimming it to focus on the specific section you want to analyze."
        },
        {
          id: "browser-compatibility",
          question: "Which browsers are supported?",
          answer: "DeepGuard AI works best on modern browsers including Chrome, Firefox, Safari, and Edge. We recommend keeping your browser updated to the latest version for optimal performance."
        }
      ]
    }
  ];

  // Getting Started steps
  const gettingStartedSteps = [
    {
      title: "Create an Account",
      description: "Sign up for a free account to access the basic features of DeepGuard AI.",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
    },
    {
      title: "Upload a Video",
      description: "Go to your Dashboard and click 'Upload Video' or drag and drop your file.",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>
    },
    {
      title: "Wait for Analysis",
      description: "Our AI system will analyze your video and look for signs of manipulation.",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
    },
    {
      title: "Review Results",
      description: "Explore the detailed analysis including confidence score and identified manipulation techniques.",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
    },
    {
      title: "Generate Report",
      description: "Export your findings as a detailed report for documentation or sharing.",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
    }
  ];

  // Support resources
  const supportResources = [
    {
      title: "Video Tutorials",
      description: "Watch step-by-step guides on how to use all features",
      link: "#tutorials",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
    },
    {
      title: "Knowledge Base",
      description: "Detailed articles on common questions and features",
      link: "#knowledge-base",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
    },
    {
      title: "Community Forum",
      description: "Connect with other users and share tips",
      link: "#forum",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    },
    {
      title: "API Documentation",
      description: "Technical guides for developers and integrations",
      link: "#api-docs",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary"><path d="M18 10h-4V4h-4v6H6l6 6 6-6zm-8 8v2h8v-2h-8z"/></svg>
    }
  ];

  // Handle contact form
  const handleContactSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Support request submitted",
      description: "We've received your message and will respond shortly."
    });
    // Reset form (in a real app, you'd clear the form fields here)
  };

  // Filter FAQs based on search query
  const filteredFAQs = searchQuery.trim() === "" 
    ? faqCategories 
    : faqCategories.map(category => ({
        ...category,
        questions: category.questions.filter(q => 
          q.question.toLowerCase().includes(searchQuery.toLowerCase()) || 
          q.answer.toLowerCase().includes(searchQuery.toLowerCase())
        )
      })).filter(category => category.questions.length > 0);

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold">Help Center</h1>
          <p className="text-muted-foreground">Find answers, learn about features, and get support</p>
        </div>

        {/* Search Bar */}
        <div className="relative mb-8">
          <Input
            type="text"
            placeholder="Search for help topics..."
            className="w-full max-w-2xl pl-12"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground"
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
        </div>

        <Tabs defaultValue="faq" className="w-full">
          <TabsList className="mb-6 w-full md:w-auto">
            <TabsTrigger value="faq">FAQ</TabsTrigger>
            <TabsTrigger value="getting-started">Getting Started</TabsTrigger>
            <TabsTrigger value="resources">Resources</TabsTrigger>
            <TabsTrigger value="contact">Contact Support</TabsTrigger>
          </TabsList>
          
          {/* FAQ Tab */}
          <TabsContent value="faq">
            <div className="grid grid-cols-1 gap-6">
              {filteredFAQs.length > 0 ? (
                filteredFAQs.map(category => (
                  category.questions.length > 0 && (
                    <Card key={category.id}>
                      <CardHeader>
                        <CardTitle>{category.name}</CardTitle>
                        <CardDescription>
                          Frequently asked questions about {category.name.toLowerCase()}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <Accordion type="single" collapsible className="w-full">
                          {category.questions.map(faq => (
                            <AccordionItem key={faq.id} value={faq.id}>
                              <AccordionTrigger className="text-left">
                                {faq.question}
                              </AccordionTrigger>
                              <AccordionContent>
                                <p className="text-muted-foreground">
                                  {faq.answer}
                                </p>
                              </AccordionContent>
                            </AccordionItem>
                          ))}
                        </Accordion>
                      </CardContent>
                    </Card>
                  )
                ))
              ) : (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground mb-4">
                      <circle cx="11" cy="11" r="8"/>
                      <path d="m21 21-4.3-4.3"/>
                    </svg>
                    <h3 className="text-xl font-medium mb-2">No results found</h3>
                    <p className="text-muted-foreground text-center max-w-md">
                      We couldn't find any FAQ that matches your search. Try using different keywords or browse through our categories.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>
          
          {/* Getting Started Tab */}
          <TabsContent value="getting-started">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Card className="mb-6">
                  <CardHeader>
                    <CardTitle>Getting Started with DeepGuard AI</CardTitle>
                    <CardDescription>
                      Follow these steps to start detecting deepfakes
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-8">
                      {gettingStartedSteps.map((step, index) => (
                        <div key={index} className="flex gap-4">
                          <div className="flex-shrink-0 h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                            {step.icon}
                          </div>
                          <div>
                            <h3 className="text-lg font-medium mb-1">
                              <span className="text-primary mr-2">{index + 1}.</span> {step.title}
                            </h3>
                            <p className="text-muted-foreground">{step.description}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Video Walkthrough</CardTitle>
                    <CardDescription>
                      Watch our quick tutorial video
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="aspect-video bg-muted rounded-md flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground">
                        <circle cx="12" cy="12" r="10"/>
                        <polygon points="10 8 16 12 10 16 10 8"/>
                      </svg>
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              <div>
                <Card className="mb-6">
                  <CardHeader>
                    <CardTitle>Understanding Your First Analysis</CardTitle>
                    <CardDescription>
                      How to interpret the detection results
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <h3 className="text-md font-medium">Confidence Score</h3>
                      <div className="flex items-center gap-4">
                        <div className="h-5 flex-grow bg-muted rounded-full overflow-hidden">
                          <div className="bg-primary h-full" style={{ width: '85%' }}></div>
                        </div>
                        <span className="text-sm font-medium">85%</span>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        The confidence score indicates how certain our AI is about the detection result. Higher scores mean greater confidence.
                      </p>
                    </div>
                    
                    <Separator />
                    
                    <div className="space-y-2">
                      <h3 className="text-md font-medium">Detected Issues</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        <div className="flex items-center gap-2 bg-muted/50 p-2 rounded-md">
                          <Badge variant="default">Face</Badge>
                          <span className="text-sm">Inconsistent blinking</span>
                        </div>
                        <div className="flex items-center gap-2 bg-muted/50 p-2 rounded-md">
                          <Badge variant="default">Audio</Badge>
                          <span className="text-sm">Voice modulation</span>
                        </div>
                        <div className="flex items-center gap-2 bg-muted/50 p-2 rounded-md">
                          <Badge variant="default">Motion</Badge>
                          <span className="text-sm">Unnatural movement</span>
                        </div>
                        <div className="flex items-center gap-2 bg-muted/50 p-2 rounded-md">
                          <Badge variant="default">Lighting</Badge>
                          <span className="text-sm">Shadow inconsistencies</span>
                        </div>
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div className="space-y-2">
                      <h3 className="text-md font-medium">Timeline Markers</h3>
                      <div className="h-6 bg-muted rounded-full relative">
                        <div className="absolute h-6 w-1 bg-red-500 rounded-full" style={{ left: '20%' }}></div>
                        <div className="absolute h-6 w-1 bg-yellow-500 rounded-full" style={{ left: '45%' }}></div>
                        <div className="absolute h-6 w-1 bg-red-500 rounded-full" style={{ left: '70%' }}></div>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Timeline markers highlight specific moments in the video where manipulation was detected.
                      </p>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Common Terms</CardTitle>
                    <CardDescription>
                      Glossary of deepfake detection terminology
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <p className="font-medium">Deepfake</p>
                        <p className="text-sm text-muted-foreground">Synthetic media where a person's likeness is replaced with someone else's using AI techniques.</p>
                      </div>
                      <div>
                        <p className="font-medium">GAN</p>
                        <p className="text-sm text-muted-foreground">Generative Adversarial Network - an AI architecture commonly used to create deepfakes.</p>
                      </div>
                      <div>
                        <p className="font-medium">Face Swapping</p>
                        <p className="text-sm text-muted-foreground">A technique that replaces one person's face with another in videos or images.</p>
                      </div>
                      <div>
                        <p className="font-medium">Voice Synthesis</p>
                        <p className="text-sm text-muted-foreground">Technology that creates artificial speech that mimics a real person's voice.</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
          
          {/* Resources Tab */}
          <TabsContent value="resources">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Support Resources</CardTitle>
                  <CardDescription>
                    Access guides and documentation to help you use DeepGuard AI
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    {supportResources.map((resource, index) => (
                      <a 
                        key={index} 
                        href={resource.link}
                        className="flex flex-col items-center bg-muted/30 p-4 rounded-lg hover:bg-muted transition-colors"
                      >
                        <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center mb-3">
                          {resource.icon}
                        </div>
                        <h3 className="text-md font-medium mb-1">{resource.title}</h3>
                        <p className="text-sm text-muted-foreground text-center">{resource.description}</p>
                      </a>
                    ))}
                  </div>
                </CardContent>
              </Card>
              
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Educational Articles</CardTitle>
                    <CardDescription>
                      Learn more about deepfakes and media manipulation
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { title: "Understanding Deepfake Technology", date: "May 10, 2025", readTime: "8 min read" },
                        { title: "How to Spot a Deepfake Video", date: "Apr 28, 2025", readTime: "6 min read" },
                        { title: "The Ethics of Synthetic Media", date: "Apr 15, 2025", readTime: "10 min read" },
                        { title: "Protecting Yourself from Disinformation", date: "Mar 30, 2025", readTime: "7 min read" }
                      ].map((article, index) => (
                        <a key={index} href="#" className="block p-4 rounded-lg hover:bg-muted transition-colors">
                          <h3 className="font-medium mb-1">{article.title}</h3>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span>{article.date}</span>
                            <span>•</span>
                            <span>{article.readTime}</span>
                          </div>
                        </a>
                      ))}
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Download Resources</CardTitle>
                    <CardDescription>
                      Helpful guides and tools
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {[
                        { title: "DeepGuard AI User Manual", size: "2.4 MB", type: "PDF" },
                        { title: "Deepfake Detection Checklist", size: "1.1 MB", type: "PDF" },
                        { title: "Media Verification Toolkit", size: "5.7 MB", type: "ZIP" },
                        { title: "Research Paper: AI Detection Methods", size: "3.2 MB", type: "PDF" }
                      ].map((download, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-md">
                          <div className="flex items-center gap-3">
                            <div className="h-8 w-8 rounded bg-primary/10 flex items-center justify-center text-primary">
                              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                                <polyline points="14 2 14 8 20 8"/>
                                <line x1="16" x2="8" y1="13" y2="13"/>
                                <line x1="16" x2="8" y1="17" y2="17"/>
                                <line x1="10" x2="8" y1="9" y2="9"/>
                              </svg>
                            </div>
                            <div>
                              <p className="font-medium text-sm">{download.title}</p>
                              <p className="text-xs text-muted-foreground">{download.size} • {download.type}</p>
                            </div>
                          </div>
                          <Button variant="ghost" size="icon">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                              <polyline points="7 10 12 15 17 10"/>
                              <line x1="12" x2="12" y1="15" y2="3"/>
                            </svg>
                          </Button>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
          
          {/* Contact Support Tab */}
          <TabsContent value="contact">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Contact Support</CardTitle>
                  <CardDescription>
                    We're here to help. Fill out the form and we'll respond as soon as possible.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleContactSubmit} className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label htmlFor="name" className="text-sm font-medium">
                          Your Name
                        </label>
                        <Input id="name" required />
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="email" className="text-sm font-medium">
                          Email Address
                        </label>
                        <Input id="email" type="email" required />
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <label htmlFor="subject" className="text-sm font-medium">
                        Subject
                      </label>
                      <Input id="subject" required />
                    </div>
                    
                    <div className="space-y-2">
                      <label htmlFor="category" className="text-sm font-medium">
                        Category
                      </label>
                      <select 
                        id="category" 
                        className="w-full p-2 rounded-md glass-dark border border-muted focus:ring-1 focus:ring-primary"
                        required
                      >
                        <option value="">Select a category</option>
                        <option value="account">Account Issues</option>
                        <option value="billing">Billing & Payments</option>
                        <option value="technical">Technical Support</option>
                        <option value="feature">Feature Request</option>
                        <option value="other">Other</option>
                      </select>
                    </div>
                    
                    <div className="space-y-2">
                      <label htmlFor="message" className="text-sm font-medium">
                        Message
                      </label>
                      <textarea 
                        id="message" 
                        rows={6} 
                        className="w-full p-2 rounded-md glass-dark border border-muted focus:ring-1 focus:ring-primary"
                        required
                      ></textarea>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        id="attach-files" 
                        className="h-4 w-4 rounded border-gray-300"
                      />
                      <label htmlFor="attach-files" className="text-sm text-muted-foreground">
                        I'd like to attach files (screenshots, videos, etc.)
                      </label>
                    </div>
                    
                    <Button type="submit" className="w-full">
                      Submit Support Request
                    </Button>
                  </form>
                </CardContent>
              </Card>
              
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Other Ways to Get Help</CardTitle>
                    <CardDescription>
                      Alternative support channels
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex items-start gap-4">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
                        </svg>
                      </div>
                      <div>
                        <h3 className="font-medium mb-1">Phone Support</h3>
                        <p className="text-sm text-muted-foreground mb-2">
                          Available Monday-Friday, 9am-5pm EST
                        </p>
                        <p className="text-sm font-medium">+1 (800) 555-1234</p>
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div className="flex items-start gap-4">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                        </svg>
                      </div>
                      <div>
                        <h3 className="font-medium mb-1">Live Chat</h3>
                        <p className="text-sm text-muted-foreground mb-2">
                          Chat with our support team in real-time
                        </p>
                        <Button variant="outline" size="sm">
                          Start Chat
                        </Button>
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div className="flex items-start gap-4">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                          <polyline points="22,6 12,13 2,6"/>
                        </svg>
                      </div>
                      <div>
                        <h3 className="font-medium mb-1">Email Support</h3>
                        <p className="text-sm text-muted-foreground mb-2">
                          We typically respond within 24 hours
                        </p>
                        <p className="text-sm font-medium">support@deepguardai.com</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Frequently Asked Support Questions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Accordion type="single" collapsible className="w-full">
                      <AccordionItem value="billing-faq">
                        <AccordionTrigger className="text-left">
                          How can I change my subscription plan?
                        </AccordionTrigger>
                        <AccordionContent>
                          <p className="text-muted-foreground">
                            You can change your subscription plan at any time by going to your Account Settings page and selecting the "Billing" tab. From there, you can upgrade, downgrade, or cancel your subscription.
                          </p>
                        </AccordionContent>
                      </AccordionItem>
                      <AccordionItem value="refund-faq">
                        <AccordionTrigger className="text-left">
                          What is your refund policy?
                        </AccordionTrigger>
                        <AccordionContent>
                          <p className="text-muted-foreground">
                            We offer a 30-day money-back guarantee for all subscription plans. If you're not satisfied with our service within the first 30 days, you can request a full refund by contacting our support team.
                          </p>
                        </AccordionContent>
                      </AccordionItem>
                      <AccordionItem value="account-faq">
                        <AccordionTrigger className="text-left">
                          I forgot my password. How do I reset it?
                        </AccordionTrigger>
                        <AccordionContent>
                          <p className="text-muted-foreground">
                            You can reset your password by clicking the "Forgot Password" link on the login page. We'll send you an email with instructions to create a new password.
                          </p>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}