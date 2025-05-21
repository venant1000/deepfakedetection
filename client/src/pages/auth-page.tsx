import { useAuth } from "@/hooks/use-auth";
import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import Navigation from "@/components/layout/navigation";

export default function AuthPage() {
  const { user, loginMutation, registerMutation } = useAuth();
  const [, navigate] = useLocation();
  const [authTab, setAuthTab] = useState("login");
  
  // Form state
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [formError, setFormError] = useState("");
  
  // Redirect to dashboard if already logged in
  useEffect(() => {
    if (user) {
      navigate("/dashboard");
    }
  }, [user, navigate]);
  
  const handleLoginSubmit = (e) => {
    e.preventDefault();
    setFormError("");
    
    if (username.length < 3) {
      setFormError("Username must be at least 3 characters");
      return;
    }
    
    if (password.length < 6) {
      setFormError("Password must be at least 6 characters");
      return;
    }
    
    loginMutation.mutate({ username, password });
  };
  
  const handleRegisterSubmit = (e) => {
    e.preventDefault();
    setFormError("");
    
    if (username.length < 3) {
      setFormError("Username must be at least 3 characters");
      return;
    }
    
    if (password.length < 6) {
      setFormError("Password must be at least 6 characters");
      return;
    }
    
    if (password !== confirmPassword) {
      setFormError("Passwords don't match");
      return;
    }
    
    registerMutation.mutate({ username, password });
  };

  return (
    <div className="pt-32 pb-20 px-6 relative grid-bg min-h-screen">
      <Navigation />
      <div className="container mx-auto max-w-6xl">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          {/* Auth Form */}
          <div className="glass rounded-xl p-8 shadow-xl border border-white/5 relative overflow-hidden">
            {/* Top accent */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary to-secondary"></div>

            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold mb-2">
                {authTab === "login" ? "Welcome Back" : "Create Account"}
              </h2>
              <p className="text-muted-foreground">
                {authTab === "login"
                  ? "Log in to your DeepGuard AI account"
                  : "Join DeepGuard AI to detect deepfakes"}
              </p>
            </div>

            <div className="mb-6">
              <div className="flex items-center mb-6">
                <div className="flex-grow h-px bg-muted"></div>
                <span className="px-4 text-sm text-muted-foreground">
                  {authTab === "login" ? "Login with your credentials" : "Create your account"}
                </span>
                <div className="flex-grow h-px bg-muted"></div>
              </div>
            </div>

            {formError && (
              <div className="bg-destructive/10 text-destructive p-3 rounded-md mb-4 text-sm">
                {formError}
              </div>
            )}

            {authTab === "login" ? (
              <form onSubmit={handleLoginSubmit} className="space-y-5">
                <div className="space-y-2">
                  <label htmlFor="username" className="block text-sm font-medium">
                    Username
                  </label>
                  <input
                    type="text"
                    id="username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="w-full p-2 rounded-md glass-dark border border-muted focus:ring-1 focus:ring-primary"
                    placeholder="Enter your username"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <label htmlFor="password" className="block text-sm font-medium">
                      Password
                    </label>
                    <a
                      href="#forgot-password"
                      className="text-sm text-primary hover:underline"
                    >
                      Forgot?
                    </a>
                  </div>
                  <input
                    type="password"
                    id="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full p-2 rounded-md glass-dark border border-muted focus:ring-1 focus:ring-primary"
                    placeholder="Enter your password"
                  />
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="rememberMe"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300"
                  />
                  <label
                    htmlFor="rememberMe"
                    className="ml-2 block text-sm text-muted-foreground"
                  >
                    Remember me
                  </label>
                </div>

                <Button
                  type="submit"
                  className="w-full py-6 bg-gradient-to-r from-primary to-secondary text-black font-semibold hover:opacity-90 transition-all"
                  disabled={loginMutation.isPending}
                >
                  {loginMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Logging in...
                    </>
                  ) : (
                    "Log In"
                  )}
                </Button>

                <p className="text-center text-sm text-muted-foreground">
                  Don't have an account?{" "}
                  <button
                    type="button"
                    onClick={() => {
                      setAuthTab("register");
                      setFormError("");
                    }}
                    className="text-primary hover:underline"
                  >
                    Sign up
                  </button>
                </p>
              </form>
            ) : (
              <form onSubmit={handleRegisterSubmit} className="space-y-5">
                <div className="space-y-2">
                  <label htmlFor="reg-username" className="block text-sm font-medium">
                    Username
                  </label>
                  <input
                    type="text"
                    id="reg-username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="w-full p-2 rounded-md glass-dark border border-muted focus:ring-1 focus:ring-primary"
                    placeholder="Choose a username"
                  />
                </div>

                <div className="space-y-2">
                  <label htmlFor="reg-password" className="block text-sm font-medium">
                    Password
                  </label>
                  <input
                    type="password"
                    id="reg-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full p-2 rounded-md glass-dark border border-muted focus:ring-1 focus:ring-primary"
                    placeholder="Create a password"
                  />
                </div>

                <div className="space-y-2">
                  <label htmlFor="confirm-password" className="block text-sm font-medium">
                    Confirm Password
                  </label>
                  <input
                    type="password"
                    id="confirm-password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full p-2 rounded-md glass-dark border border-muted focus:ring-1 focus:ring-primary"
                    placeholder="Confirm your password"
                  />
                </div>

                <Button
                  type="submit"
                  className="w-full py-6 bg-gradient-to-r from-primary to-secondary text-black font-semibold hover:opacity-90 transition-all"
                  disabled={registerMutation.isPending}
                >
                  {registerMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating account...
                    </>
                  ) : (
                    "Create Account"
                  )}
                </Button>

                <p className="text-center text-sm text-muted-foreground">
                  Already have an account?{" "}
                  <button
                    type="button"
                    onClick={() => {
                      setAuthTab("login");
                      setFormError("");
                    }}
                    className="text-primary hover:underline"
                  >
                    Log in
                  </button>
                </p>
              </form>
            )}
          </div>

          {/* Hero section */}
          <div className="text-center md:text-left">
            <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-6">
              <span className="gradient-text">Detect Deepfakes</span>
              <br />
              With Advanced AI
            </h1>
            <p className="text-lg text-muted-foreground mb-8 max-w-lg">
              DeepGuard AI uses Google's Gemini technology to analyze videos and detect 
              manipulation with unprecedented accuracy, protecting you from misinformation 
              in real-time.
            </p>
            <div className="glass p-6 rounded-xl max-w-md">
              <div className="flex items-center gap-4 mb-4">
                <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center text-primary">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Stay Protected</h3>
                  <p className="text-muted-foreground">
                    In a world where seeing isn't always believing
                  </p>
                </div>
              </div>
              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary mt-1"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><path d="m9 12 2 2 4-4"/></svg>
                  <span>Analyze suspicious videos with one click</span>
                </li>
                <li className="flex items-start gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary mt-1"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><path d="m9 12 2 2 4-4"/></svg>
                  <span>Get detailed reports on manipulated content</span>
                </li>
                <li className="flex items-start gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary mt-1"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><path d="m9 12 2 2 4-4"/></svg>
                  <span>Educate yourself with our AI assistant</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
