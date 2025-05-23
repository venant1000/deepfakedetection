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
      name: "Account Information",
      questions: [
        {
          id: "account-features",
          question: "What features are available in my account?",
          answer: "All accounts have access to our full range of features including video analysis, detailed reports, history tracking, and educational resources on deepfake detection."
        },
        {
          id: "account-limits",
          question: "Are there any usage limits?",
          answer: "Currently, there are no strict limits on video analysis. We recommend keeping videos under 10 minutes in length for optimal performance and faster processing times."
        },
        {
          id: "account-inactive",
          question: "What happens to inactive accounts?",
          answer: "Accounts that remain inactive for more than 12 months may be archived for security purposes. You can always log back in to reactivate your account and access your analysis history."
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
          answer: "Yes, the maximum file size is 500MB for all users. If your video exceeds this limit, consider compressing it or trimming it to focus on the specific section you want to analyze."
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
                    <CardTitle>Quick Tips for Better Results</CardTitle>
                    <CardDescription>
                      Optimize your deepfake detection experience
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-4">
                      <li className="flex items-start gap-2">
                        <div className="h-5 w-5 rounded-full bg-primary/10 flex-shrink-0 flex items-center justify-center mt-0.5">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        </div>
                        <p className="text-sm">
                          <span className="font-medium">Use high-quality videos</span> - Higher quality videos contain more data for our AI to analyze, leading to more accurate results.
                        </p>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="h-5 w-5 rounded-full bg-primary/10 flex-shrink-0 flex items-center justify-center mt-0.5">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        </div>
                        <p className="text-sm">
                          <span className="font-medium">Focus on faces</span> - Most deepfakes manipulate facial features, so videos with clear views of faces provide better analysis.
                        </p>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="h-5 w-5 rounded-full bg-primary/10 flex-shrink-0 flex items-center justify-center mt-0.5">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        </div>
                        <p className="text-sm">
                          <span className="font-medium">Check both visuals and audio</span> - Our system analyzes both video and audio components to detect inconsistencies.
                        </p>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="h-5 w-5 rounded-full bg-primary/10 flex-shrink-0 flex items-center justify-center mt-0.5">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        </div>
                        <p className="text-sm">
                          <span className="font-medium">Review the detailed report</span> - Don't just check the verdict; explore the detailed findings for a complete understanding of potential issues.
                        </p>
                      </li>
                    </ul>
                  </CardContent>
                </Card>
              </div>
              
              <div>
                <Card className="mb-6">
                  <CardHeader>
                    <CardTitle>Understanding Analysis Results</CardTitle>
                    <CardDescription>
                      Learn how to interpret our deepfake detection findings
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div>
                      <h3 className="font-medium mb-2">Confidence Score</h3>
                      <div className="flex items-center gap-2 mb-1">
                        <Badge className="bg-green-500">High (90-100%)</Badge>
                        <span className="text-sm">Very reliable detection</span>
                      </div>
                      <div className="flex items-center gap-2 mb-1">
                        <Badge className="bg-yellow-500">Medium (70-89%)</Badge>
                        <span className="text-sm">Fairly reliable detection</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className="bg-red-500">Low (Below 70%)</Badge>
                        <span className="text-sm">Requires manual review</span>
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div>
                      <h3 className="font-medium mb-2">Common Detection Indicators</h3>
                      <ul className="space-y-2 text-sm">
                        <li className="flex gap-2">
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <rect x="2" y="2" width="20" height="20" rx="5" ry="5"/>
                            <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/>
                            <line x1="17.5" y1="6.5" x2="17.51" y2="6.5"/>
                          </svg>
                          <span>Facial inconsistencies (blurring, unnatural features)</span>
                        </li>
                        <li className="flex gap-2">
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <path d="M17.5 8a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"/>
                            <path d="M6.5 21a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z"/>
                            <path d="m15 5-2 12-2 4"/>
                            <path d="m18 5-4.5 16.5"/>
                            <path d="m13.5 17.5-7 1.5"/>
                          </svg>
                          <span>Unnatural body movements or postures</span>
                        </li>
                        <li className="flex gap-2">
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <path d="M9.6 14H4a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2h3.6a2 2 0 0 1 1.4.6L10.4 5"/>
                            <path d="M20 15a2 2 0 0 1-2 2h-9.6a2 2 0 0 1-1.4-.6L5.6 15"/>
                            <path d="M22 5a2 2 0 0 0-2-2h-3.6a2 2 0 0 0-1.4.6L13.6 5"/>
                            <path d="M14.4 10H20a2 2 0 0 1 2 2v7c0 1.1-.9 2-2 2h-3.6a2 2 0 0 1-1.4-.6L13.6 19"/>
                          </svg>
                          <span>Audio-visual misalignment or unnatural speech</span>
                        </li>
                        <li className="flex gap-2">
                          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <circle cx="12" cy="12" r="4"/>
                            <path d="M12 4a1 1 0 0 0-1 1v2a1 1 0 0 0 2 0V5a1 1 0 0 0-1-1Z"/>
                            <path d="M17.66 6.345a1 1 0 0 0-1.41 0l-1.415 1.413a1 1 0 1 0 1.41 1.42l1.415-1.414a1 1 0 0 0 0-1.42Z"/>
                            <path d="M20 11a1 1 0 0 0-1-1h-2a1 1 0 0 0 0 2h2a1 1 0 0 0 1-1Z"/>
                            <path d="M17.66 15.655a1 1 0 0 0 0 1.414l1.414 1.414a1 1 0 0 0 1.42-1.414l-1.415-1.414a1 1 0 0 0-1.42 0Z"/>
                            <path d="M12 17a1 1 0 0 0-1 1v2a1 1 0 0 0 2 0v-2a1 1 0 0 0-1-1Z"/>
                            <path d="M7.76 15.655a1 1 0 0 0-1.42 0l-1.414 1.414a1 1 0 1 0 1.414 1.414l1.414-1.414a1 1 0 0 0 0-1.414Z"/>
                            <path d="M6 11a1 1 0 0 0-1-1H3a1 1 0 0 0 0 2h2a1 1 0 0 0 1-1Z"/>
                            <path d="M7.76 6.345a1 1 0 0 0-1.42 1.42l1.414 1.414a1 1 0 1 0 1.414-1.42L7.759 6.344Z"/>
                          </svg>
                          <span>Inconsistent lighting or shadows</span>
                        </li>
                      </ul>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary mr-2">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                      </svg>
                      Staying Protected
                    </CardTitle>
                    <CardDescription>
                      Tips for protecting yourself from deepfake threats
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-4">
                      <li className="flex items-start gap-2">
                        <div className="h-5 w-5 rounded-full bg-primary/10 flex-shrink-0 flex items-center justify-center mt-0.5">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        </div>
                        <p className="text-sm">Verify information from multiple trusted sources before believing or sharing videos.</p>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="h-5 w-5 rounded-full bg-primary/10 flex-shrink-0 flex items-center justify-center mt-0.5">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        </div>
                        <p className="text-sm">Be cautious of videos showing people saying or doing things that seem out of character.</p>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="h-5 w-5 rounded-full bg-primary/10 flex-shrink-0 flex items-center justify-center mt-0.5">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        </div>
                        <p className="text-sm">Check video metadata and source information when possible.</p>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="h-5 w-5 rounded-full bg-primary/10 flex-shrink-0 flex items-center justify-center mt-0.5">
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                        </div>
                        <p className="text-sm">Use DeepGuard AI regularly to scan suspicious media before sharing or making important decisions based on it.</p>
                      </li>
                    </ul>
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