import { useState } from "react";
import { useLocation } from "wouter";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/sheet";

export default function Navigation() {
  const [, navigate] = useLocation();
  const { user } = useAuth();
  const [isOpen, setIsOpen] = useState(false);

  const navItems = [
    { label: "Features", href: "/#features" },
    { label: "How It Works", href: "/#how-it-works" },
    { label: "Pricing", href: "/#pricing" },
    { label: "About", href: "/#about" }
  ];

  return (
    <nav className="glass-dark fixed w-full top-0 z-50 py-4">
      <div className="container mx-auto px-6 flex justify-between items-center">
        <div 
          className="flex items-center cursor-pointer" 
          onClick={() => navigate("/")}
        >
          <div className="h-8 w-8 rounded-md bg-gradient-to-br from-primary to-secondary flex items-center justify-center mr-3">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/></svg>
          </div>
          <span className="text-xl font-bold">DeepGuard<span className="text-primary">AI</span></span>
        </div>
        
        <div className="hidden md:flex space-x-8 items-center">
          {navItems.map((item, index) => (
            <a 
              key={index} 
              href={item.href} 
              className="text-muted-foreground hover:text-primary transition-colors"
            >
              {item.label}
            </a>
          ))}
          
          {user ? (
            <Button
              className="py-2 px-4 rounded-lg bg-gradient-to-r from-primary to-secondary hover:opacity-90 button-glow transition-all"
              onClick={() => navigate("/dashboard")}
            >
              Dashboard
            </Button>
          ) : (
            <>
              <Button
                variant="outline"
                className="py-2 px-4 rounded-lg border border-primary text-primary hover:bg-primary hover:text-black transition-all"
                onClick={() => navigate("/auth")}
              >
                Login
              </Button>
              <Button
                className="py-2 px-4 rounded-lg bg-gradient-to-r from-primary to-secondary hover:opacity-90 button-glow transition-all"
                onClick={() => navigate("/auth")}
              >
                Sign Up
              </Button>
            </>
          )}
        </div>
        
        <div className="md:hidden">
          <Sheet open={isOpen} onOpenChange={setIsOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/></svg>
              </Button>
            </SheetTrigger>
            <SheetContent className="bg-background/95 backdrop-blur-md border-muted">
              <div className="flex flex-col space-y-6 mt-8">
                {navItems.map((item, index) => (
                  <a 
                    key={index} 
                    href={item.href} 
                    className="text-foreground text-lg py-2"
                    onClick={() => setIsOpen(false)}
                  >
                    {item.label}
                  </a>
                ))}
                
                <div className="flex flex-col gap-4 pt-4">
                  {user ? (
                    <Button
                      className="w-full py-6 bg-gradient-to-r from-primary to-secondary hover:opacity-90 button-glow transition-all"
                      onClick={() => {
                        setIsOpen(false);
                        navigate("/dashboard");
                      }}
                    >
                      Dashboard
                    </Button>
                  ) : (
                    <>
                      <Button
                        variant="outline"
                        className="w-full py-6 border border-primary text-primary hover:bg-primary hover:text-black transition-all"
                        onClick={() => {
                          setIsOpen(false);
                          navigate("/auth");
                        }}
                      >
                        Login
                      </Button>
                      <Button
                        className="w-full py-6 bg-gradient-to-r from-primary to-secondary hover:opacity-90 button-glow transition-all"
                        onClick={() => {
                          setIsOpen(false);
                          navigate("/auth");
                        }}
                      >
                        Sign Up
                      </Button>
                    </>
                  )}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </nav>
  );
}
