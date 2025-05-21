import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";

export default function ProfilePage() {
  const { user } = useAuth();
  const { toast } = useToast();
  
  const [profileForm, setProfileForm] = useState({
    username: user?.username || "",
    email: user?.email || "",
    fullName: user?.fullName || "",
    organization: user?.organization || "",
  });
  
  const [securityForm, setSecurityForm] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  });
  
  const [notificationSettings, setNotificationSettings] = useState({
    emailAlerts: true,
    desktopNotifications: true,
    weeklyReports: false,
    successfulAnalysis: true,
    potentialDeepfakes: true,
  });
  
  const handleProfileUpdate = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Simulate API call
    setTimeout(() => {
      toast({
        title: "Profile updated",
        description: "Your profile information has been updated successfully.",
      });
    }, 500);
  };
  
  const handlePasswordUpdate = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (securityForm.newPassword !== securityForm.confirmPassword) {
      toast({
        title: "Passwords don't match",
        description: "New password and confirmation password must match.",
        variant: "destructive",
      });
      return;
    }
    
    // Simulate API call
    setTimeout(() => {
      toast({
        title: "Password updated",
        description: "Your password has been changed successfully.",
      });
      
      setSecurityForm({
        currentPassword: "",
        newPassword: "",
        confirmPassword: "",
      });
    }, 500);
  };
  
  const handleNotificationToggle = (setting: string) => {
    setNotificationSettings(prev => ({
      ...prev,
      [setting]: !prev[setting as keyof typeof prev],
    }));
    
    // Simulate API call
    toast({
      title: "Notification settings updated",
      description: "Your notification preferences have been saved.",
    });
  };

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isAdmin={user?.role === "admin"} />
      
      <div className="flex-1 ml-20 md:ml-64 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-8">Profile Settings</h1>
          
          <Tabs defaultValue="profile" className="w-full">
            <TabsList className="grid w-full grid-cols-3 mb-8">
              <TabsTrigger value="profile">Profile Information</TabsTrigger>
              <TabsTrigger value="security">Security</TabsTrigger>
              <TabsTrigger value="notifications">Notifications</TabsTrigger>
            </TabsList>
            
            <TabsContent value="profile">
              <div className="glass rounded-xl p-8">
                <form onSubmit={handleProfileUpdate}>
                  <div className="flex items-center gap-6 mb-8">
                    <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center text-primary text-xl font-semibold uppercase">
                      {user?.username.substring(0, 2) || "U"}
                    </div>
                    
                    <div>
                      <h2 className="text-xl font-semibold">{user?.username}</h2>
                      <p className="text-muted-foreground">
                        User Role: {user?.role === "admin" ? "Administrator" : "Standard User"}
                      </p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    <div className="space-y-2">
                      <Label htmlFor="username">Username</Label>
                      <Input
                        id="username"
                        value={profileForm.username}
                        onChange={(e) => setProfileForm({...profileForm, username: e.target.value})}
                        disabled
                      />
                      <p className="text-xs text-muted-foreground">Username cannot be changed</p>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="email">Email Address</Label>
                      <Input
                        id="email"
                        type="email"
                        value={profileForm.email}
                        onChange={(e) => setProfileForm({...profileForm, email: e.target.value})}
                        placeholder="your.email@example.com"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="fullName">Full Name</Label>
                      <Input
                        id="fullName"
                        value={profileForm.fullName}
                        onChange={(e) => setProfileForm({...profileForm, fullName: e.target.value})}
                        placeholder="Enter your full name"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="organization">Organization</Label>
                      <Input
                        id="organization"
                        value={profileForm.organization}
                        onChange={(e) => setProfileForm({...profileForm, organization: e.target.value})}
                        placeholder="Your company or organization"
                      />
                    </div>
                  </div>
                  
                  <Button type="submit">Save Changes</Button>
                </form>
              </div>
            </TabsContent>
            
            <TabsContent value="security">
              <div className="glass rounded-xl p-8">
                <h2 className="text-xl font-semibold mb-6">Change Password</h2>
                
                <form onSubmit={handlePasswordUpdate} className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="currentPassword">Current Password</Label>
                    <Input
                      id="currentPassword"
                      type="password"
                      value={securityForm.currentPassword}
                      onChange={(e) => setSecurityForm({...securityForm, currentPassword: e.target.value})}
                      placeholder="Enter your current password"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="newPassword">New Password</Label>
                    <Input
                      id="newPassword"
                      type="password"
                      value={securityForm.newPassword}
                      onChange={(e) => setSecurityForm({...securityForm, newPassword: e.target.value})}
                      placeholder="Enter your new password"
                    />
                    <p className="text-xs text-muted-foreground">
                      Password must be at least 8 characters long and include a mix of letters, numbers, and symbols.
                    </p>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword">Confirm New Password</Label>
                    <Input
                      id="confirmPassword"
                      type="password"
                      value={securityForm.confirmPassword}
                      onChange={(e) => setSecurityForm({...securityForm, confirmPassword: e.target.value})}
                      placeholder="Confirm your new password"
                    />
                  </div>
                  
                  <Button type="submit">Update Password</Button>
                </form>
                
                <div className="mt-12 pt-8 border-t border-muted">
                  <h2 className="text-xl font-semibold mb-6">Account Security</h2>
                  
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className="font-medium">Two-Factor Authentication</p>
                      <p className="text-sm text-muted-foreground">Add an extra layer of security to your account</p>
                    </div>
                    <Button variant="outline">Enable</Button>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Active Sessions</p>
                      <p className="text-sm text-muted-foreground">Manage your active login sessions</p>
                    </div>
                    <Button variant="outline">View</Button>
                  </div>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="notifications">
              <div className="glass rounded-xl p-8">
                <h2 className="text-xl font-semibold mb-6">Notification Preferences</h2>
                
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Email Alerts</p>
                      <p className="text-sm text-muted-foreground">Receive notifications via email</p>
                    </div>
                    <Switch 
                      checked={notificationSettings.emailAlerts} 
                      onCheckedChange={() => handleNotificationToggle("emailAlerts")}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Desktop Notifications</p>
                      <p className="text-sm text-muted-foreground">Show notifications in your browser</p>
                    </div>
                    <Switch 
                      checked={notificationSettings.desktopNotifications} 
                      onCheckedChange={() => handleNotificationToggle("desktopNotifications")}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Weekly Summary Reports</p>
                      <p className="text-sm text-muted-foreground">Receive a weekly summary of your account activity</p>
                    </div>
                    <Switch 
                      checked={notificationSettings.weeklyReports} 
                      onCheckedChange={() => handleNotificationToggle("weeklyReports")}
                    />
                  </div>
                  
                  <div className="pt-6 border-t border-muted">
                    <h3 className="text-lg font-medium mb-4">Notify me about:</h3>
                    
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <p>Successful video analysis</p>
                        <Switch 
                          checked={notificationSettings.successfulAnalysis} 
                          onCheckedChange={() => handleNotificationToggle("successfulAnalysis")}
                        />
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <p>Potential deepfakes detected</p>
                        <Switch 
                          checked={notificationSettings.potentialDeepfakes} 
                          onCheckedChange={() => handleNotificationToggle("potentialDeepfakes")}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}