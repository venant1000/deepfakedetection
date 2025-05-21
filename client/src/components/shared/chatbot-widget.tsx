import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";

interface Message {
  sender: "bot" | "user";
  text: string;
  isMarkdown?: boolean;
}

export default function ChatbotWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: "bot",
      text: "Hi there! I'm your DeepGuard AI assistant. How can I help you learn about deepfakes today?"
    }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messageContainerRef = useRef<HTMLDivElement>(null);

  const suggestions = [
    "How do deepfakes work?",
    "Audio deepfakes",
    "Deepfake laws"
  ];

  useEffect(() => {
    if (messageContainerRef.current) {
      messageContainerRef.current.scrollTop = messageContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = () => {
    if (input.trim() === "") return;
    
    // Add user message
    const userMessage: Message = { sender: "user", text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    
    // Simulate bot typing
    setIsTyping(true);
    
    // Simulate bot response after delay
    setTimeout(() => {
      let botResponse: Message;
      
      // Simple response logic based on keywords
      const lowercaseInput = input.toLowerCase();
      
      if (lowercaseInput.includes("how") && lowercaseInput.includes("spot")) {
        botResponse = {
          sender: "bot",
          text: "Here are some key signs to look for when trying to identify a deepfake:\n\n* Unnatural eye movements or lack of blinking\n* Inconsistent lighting on the face\n* Poor lip synchronization with audio\n* Blurry or changing face edges\n* Unnatural skin tone or texture\n\nWould you like me to explain any of these in more detail?",
          isMarkdown: true
        };
      } else if (lowercaseInput.includes("deepfake") && lowercaseInput.includes("work")) {
        botResponse = {
          sender: "bot",
          text: "Deepfakes work by using artificial intelligence, specifically deep learning algorithms, to replace a person's likeness in an image or video with someone else's. The technology typically uses two AI models working together:\n\n1. **Encoder**: Analyzes and learns patterns from both face images\n2. **Decoder**: Reconstructs faces with swapped features\n\nThe most common method uses Generative Adversarial Networks (GANs) where two neural networks compete - one creates the fake content, while the other tries to detect if it's real or fake.",
          isMarkdown: true
        };
      } else if (lowercaseInput.includes("audio")) {
        botResponse = {
          sender: "bot",
          text: "Audio deepfakes, also called voice cloning, use AI to synthesize a person's voice. With just a few minutes of sample audio, deepfake technology can create realistic voice replicas that can say anything. \n\nThese are created using neural networks that analyze voice patterns, intonation, and speech characteristics. Common signs of audio deepfakes include:\n\n* Unnatural rhythm or cadence\n* Inconsistent breathing patterns\n* Strange artifacts or glitches in audio\n* Unusual background noise patterns",
          isMarkdown: true
        };
      } else {
        botResponse = {
          sender: "bot",
          text: "That's an interesting question about deepfakes. Would you like me to provide more specific information about detection techniques, creation methods, or how to protect yourself from deepfake content?"
        };
      }
      
      setMessages(prev => [...prev, botResponse]);
      setIsTyping(false);
    }, 1500);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    // Submit the suggestion immediately
    const userMessage: Message = { sender: "user", text: suggestion };
    setMessages(prev => [...prev, userMessage]);
    
    // Simulate bot typing
    setIsTyping(true);
    
    // Simulate bot response after delay for suggestions
    setTimeout(() => {
      let botResponse: Message;
      
      if (suggestion === "How do deepfakes work?") {
        botResponse = {
          sender: "bot",
          text: "Deepfakes work by using artificial intelligence, specifically deep learning algorithms, to replace a person's likeness in an image or video with someone else's. The technology typically uses two AI models working together:\n\n1. **Encoder**: Analyzes and learns patterns from both face images\n2. **Decoder**: Reconstructs faces with swapped features\n\nThe most common method uses Generative Adversarial Networks (GANs) where two neural networks compete - one creates the fake content, while the other tries to detect if it's real or fake.",
          isMarkdown: true
        };
      } else if (suggestion === "Audio deepfakes") {
        botResponse = {
          sender: "bot",
          text: "Audio deepfakes, also called voice cloning, use AI to synthesize a person's voice. With just a few minutes of sample audio, deepfake technology can create realistic voice replicas that can say anything. \n\nThese are created using neural networks that analyze voice patterns, intonation, and speech characteristics. Common signs of audio deepfakes include:\n\n* Unnatural rhythm or cadence\n* Inconsistent breathing patterns\n* Strange artifacts or glitches in audio\n* Unusual background noise patterns",
          isMarkdown: true
        };
      } else if (suggestion === "Deepfake laws") {
        botResponse = {
          sender: "bot",
          text: "Deepfake regulation is still evolving globally. Some key legal developments include:\n\n* **United States**: Several states (CA, TX, VA) have laws against non-consensual deepfake pornography\n* **China**: Requires deepfakes to be clearly labeled as synthetic\n* **EU**: The Digital Services Act includes provisions that could affect deepfake distribution\n* **South Korea**: Has specific laws prohibiting the creation of deepfake content without consent\n\nMany jurisdictions are working on additional legislation as the technology advances.",
          isMarkdown: true
        };
      }
      
      setMessages(prev => [...prev, botResponse!]);
      setIsTyping(false);
    }, 1500);
    
    setInput("");
  };

  // Format markdown-like text for display
  const formatMessage = (text: string, isMarkdown?: boolean) => {
    if (!isMarkdown) return text;
    
    // Replace markdown-style formatting with HTML
    const formattedText = text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
      .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
      .split('\n\n').map(paragraph => 
        paragraph.startsWith('* ') 
          ? `<ul>${paragraph.split('\n').map(item => `<li>${item.substring(2)}</li>`).join('')}</ul>` 
          : paragraph.startsWith('1. ')
            ? `<ol>${paragraph.split('\n').map(item => `<li>${item.substring(3)}</li>`).join('')}</ol>`
            : `<p>${paragraph}</p>`
      ).join('');
    
    return formattedText;
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {isOpen && (
        <div className="glass rounded-xl w-80 shadow-xl mb-4">
          <div className="flex items-center justify-between p-4 border-b border-muted">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full flex items-center justify-center bg-primary text-black">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 8V4H8" />
                  <rect x="2" y="2" width="20" height="8" rx="2" />
                  <path d="M2 14h12" />
                  <path d="M2 20h16" />
                  <path d="M18 14h4v6h-4z" />
                </svg>
              </div>
              <div>
                <p className="font-medium">DeepGuard AI Assistant</p>
                <p className="text-xs text-muted-foreground">Powered by Gemini</p>
              </div>
            </div>
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={() => setIsOpen(false)}
              className="text-muted-foreground hover:text-white"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6 6 18" />
                <path d="m6 6 12 12" />
              </svg>
            </Button>
          </div>
          
          <div className="p-4 h-80 overflow-y-auto" ref={messageContainerRef}>
            {/* Chat messages */}
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div 
                  key={index} 
                  className={`flex items-start gap-2 ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div className={`h-8 w-8 rounded-full flex-shrink-0 flex items-center justify-center ${
                    message.sender === 'bot' 
                      ? 'bg-primary text-black' 
                      : 'bg-muted'
                  }`}>
                    {message.sender === 'bot' ? (
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 8V4H8" />
                        <rect x="2" y="2" width="20" height="8" rx="2" />
                        <path d="M2 14h12" />
                        <path d="M2 20h16" />
                        <path d="M18 14h4v6h-4z" />
                      </svg>
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
                        <circle cx="12" cy="7" r="4" />
                      </svg>
                    )}
                  </div>
                  <div className={`p-3 rounded-lg ${
                    message.sender === 'bot' 
                      ? 'glass-dark rounded-tl-none' 
                      : 'bg-secondary/20 rounded-tr-none'
                  }`}>
                    {message.isMarkdown ? (
                      <div 
                        className="text-sm prose prose-sm dark:prose-invert max-w-none"
                        dangerouslySetInnerHTML={{ __html: formatMessage(message.text, true) }} 
                      />
                    ) : (
                      <p className="text-sm">{message.text}</p>
                    )}
                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="flex items-start gap-2">
                  <div className="h-8 w-8 rounded-full flex-shrink-0 flex items-center justify-center bg-primary text-black">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 8V4H8" />
                      <rect x="2" y="2" width="20" height="8" rx="2" />
                      <path d="M2 14h12" />
                      <path d="M2 20h16" />
                      <path d="M18 14h4v6h-4z" />
                    </svg>
                  </div>
                  <div className="glass-dark p-3 rounded-lg rounded-tl-none">
                    <div className="flex space-x-2">
                      <div className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="h-2 w-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="p-4 border-t border-muted">
            {/* Quick suggestions */}
            <div className="flex flex-wrap gap-2 mb-3">
              {suggestions.map((suggestion, index) => (
                <button 
                  key={index}
                  className="py-1 px-3 rounded-full text-xs bg-muted hover:bg-muted/80 transition-colors"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
            
            {/* Input area */}
            <div className="flex gap-2">
              <input 
                type="text" 
                placeholder="Type your question..." 
                className="flex-grow py-2 px-3 rounded-lg glass-dark border border-muted focus:outline-none focus:border-primary transition-colors"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleSendMessage();
                  }
                }}
              />
              <Button 
                className="p-2 rounded-lg bg-primary text-black"
                onClick={handleSendMessage}
                disabled={input.trim() === ''}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m22 2-7 20-4-9-9-4Z" />
                  <path d="M22 2 11 13" />
                </svg>
              </Button>
            </div>
          </div>
        </div>
      )}
      
      <Button 
        className="h-14 w-14 rounded-full bg-gradient-to-r from-primary to-secondary flex items-center justify-center animate-pulse shadow-lg hover:scale-105 transition-transform"
        onClick={() => setIsOpen(!isOpen)}
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 8V4H8" />
          <rect x="2" y="2" width="20" height="8" rx="2" />
          <path d="M2 14h12" />
          <path d="M2 20h16" />
          <path d="M18 14h4v6h-4z" />
        </svg>
      </Button>
    </div>
  );
}
