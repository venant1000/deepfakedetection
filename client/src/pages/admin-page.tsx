import { useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import AdminStats from "@/components/admin/admin-stats";
import AnalyticsDashboard from "@/components/admin/analytics-dashboard";
import UserManagement from "@/components/admin/user-management";
import { useToast } from "@/hooks/use-toast";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function AdminPage() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();

  // Check if user is admin
  useEffect(() => {
    if (user && user.username !== "admin") {
      toast({
        title: "Access Denied",
        description: "You don't have permission to access this page.",
        variant: "destructive"
      });
      navigate("/dashboard");
    }
  }, [user, navigate, toast]);

  const adminSections = [
    {
      title: "Deepfake Analytics",
      description: "In-depth analysis of deepfake detection metrics",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>,
      path: "/admin/deepfake-analytics"
    },
    {
      title: "User Management",
      description: "Manage user accounts, permissions and activities",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>,
      path: "/admin/users"
    },
    {
      title: "System Logs",
      description: "View and analyze system activity and error logs",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>,
      path: "/admin/logs"
    },
    {
      title: "System Settings",
      description: "Configure platform settings and preferences",
      icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>,
      path: "/admin/settings"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Sidebar isAdmin />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Admin Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">Admin Dashboard</h1>
            <p className="text-muted-foreground">Welcome to the admin control panel</p>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="relative">
              <input
                type="text"
                placeholder="Search..."
                className="py-2 px-4 pl-10 rounded-lg glass-dark border border-muted focus:outline-none focus:border-primary transition-colors w-64"
              />
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
              >
                <circle cx="11" cy="11" r="8"/>
                <path d="m21 21-4.3-4.3"/>
              </svg>
            </div>
          </div>
        </div>

        {/* Overview Stats */}
        <AdminStats />
        
        {/* Admin Menu Cards */}
        <h2 className="text-xl font-semibold mt-8 mb-4">Admin Controls</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
          {adminSections.map((section, index) => (
            <Card key={index} className="hover:border-primary/50 transition-colors cursor-pointer">
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <CardTitle className="text-lg font-medium">{section.title}</CardTitle>
                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                  {section.icon}
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-sm min-h-[40px]">
                  {section.description}
                </CardDescription>
                <Button 
                  variant="default" 
                  className="w-full mt-4"
                  onClick={() => navigate(section.path)}
                >
                  Access
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
        
        {/* Quick help text */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Admin Dashboard</CardTitle>
            <CardDescription>Use the controls above to manage your DeepGuardAI system</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              The DeepGuardAI admin dashboard provides you with advanced tools for managing users, 
              reviewing analytics, monitoring system logs, and configuring system settings. Click on any
              of the control panels above to access these features.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}