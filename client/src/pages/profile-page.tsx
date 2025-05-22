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
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { useToast } from "@/hooks/use-toast";

export default function ProfilePage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  
  // Form state
  const [profileData, setProfileData] = useState({
    username: user?.username || "",
    fullName: "John Doe", // Mock data
    email: "user@example.com", // Mock data
    bio: "Deepfake detection enthusiast and tech advocate. Working to make the internet safer through AI-powered media analysis.",
    organization: "Media Truth Initiative",
    location: "San Francisco, CA",
    website: "https://example.com"
  });

  // Security settings
  const [securitySettings, setSecuritySettings] = useState({
    twoFactorAuth: false,
    loginNotifications: true,
    sessionTimeout: 30 // minutes
  });

  // Notification preferences
  const [notificationPrefs, setNotificationPrefs] = useState({
    emailNotifications: true,
    analysisCompletionAlerts: true,
    weeklyReportSummary: false,
    securityAlerts: true,
    marketingEmails: false
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSaveProfile = () => {
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      toast({
        title: "Profile updated",
        description: "Your profile information has been updated successfully."
      });
    }, 1000);
  };

  const handleChangePassword = () => {
    toast({
      title: "Password changed",
      description: "Your password has been updated successfully."
    });
  };

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map(n => n[0])
      .join("")
      .toUpperCase();
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Page Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">My Profile</h1>
            <p className="text-muted-foreground">Manage your account information and preferences</p>
          </div>
        </div>

        <Tabs defaultValue="profile" className="w-full">
          <TabsList className="mb-6 w-full md:w-auto">
            <TabsTrigger value="profile">Profile Information</TabsTrigger>
            <TabsTrigger value="security">Security</TabsTrigger>
            <TabsTrigger value="notifications">Notifications</TabsTrigger>
            <TabsTrigger value="api">API Access</TabsTrigger>
          </TabsList>
          
          {/* Profile Information Tab */}
          <TabsContent value="profile">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Personal Information</CardTitle>
                    <CardDescription>
                      Update your personal details and public profile
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label htmlFor="username">Username</Label>
                        <Input 
                          id="username" 
                          name="username"
                          value={profileData.username}
                          onChange={handleInputChange}
                          disabled
                        />
                        <p className="text-sm text-muted-foreground">
                          Your username cannot be changed.
                        </p>
                      </div>
                      
                      <div className="space-y-2">
                        <Label htmlFor="fullName">Full Name</Label>
                        <Input 
                          id="fullName" 
                          name="fullName"
                          value={profileData.fullName}
                          onChange={handleInputChange}
                        />
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="email">Email Address</Label>
                      <Input 
                        id="email" 
                        name="email"
                        type="email"
                        value={profileData.email}
                        onChange={handleInputChange}
                      />
                      <p className="text-sm text-muted-foreground">
                        We'll use this email for notifications and account recovery.
                      </p>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="bio">Bio</Label>
                      <textarea 
                        id="bio" 
                        name="bio"
                        value={profileData.bio}
                        onChange={handleInputChange}
                        className="w-full p-2 rounded-md glass-dark border border-muted focus:ring-1 focus:ring-primary min-h-[120px]"
                      />
                      <p className="text-sm text-muted-foreground">
                        Tell us a bit about yourself and your interest in deepfake detection.
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label htmlFor="organization">Organization</Label>
                        <Input 
                          id="organization" 
                          name="organization"
                          value={profileData.organization}
                          onChange={handleInputChange}
                        />
                      </div>
                      
                      <div className="space-y-2">
                        <Label htmlFor="location">Location</Label>
                        <Input 
                          id="location" 
                          name="location"
                          value={profileData.location}
                          onChange={handleInputChange}
                        />
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="website">Website</Label>
                      <Input 
                        id="website" 
                        name="website"
                        value={profileData.website}
                        onChange={handleInputChange}
                      />
                    </div>
                  </CardContent>
                  <CardFooter className="flex justify-end border-t pt-6">
                    <Button onClick={handleSaveProfile} disabled={isLoading}>
                      {isLoading ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Saving...
                        </>
                      ) : (
                        'Save Changes'
                      )}
                    </Button>
                  </CardFooter>
                </Card>
              </div>
              
              <div>
                <Card>
                  <CardHeader>
                    <CardTitle>Profile Picture</CardTitle>
                    <CardDescription>
                      Update your profile photo
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-col items-center">
                    <Avatar className="h-40 w-40 mb-4">
                      <AvatarImage src="" />
                      <AvatarFallback className="text-4xl bg-primary/10 text-primary">
                        {getInitials(profileData.fullName)}
                      </AvatarFallback>
                    </Avatar>
                    
                    <div className="flex gap-2 mt-4">
                      <Button variant="outline">
                        Upload New
                      </Button>
                      <Button variant="destructive">
                        Remove
                      </Button>
                    </div>
                    
                    <p className="text-xs text-muted-foreground mt-4 text-center">
                      Recommended size: 500x500 pixels.<br />
                      JPG, PNG or GIF. Max 2MB.
                    </p>
                  </CardContent>
                </Card>
                
                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle>Account Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        <div className="h-3 w-3 rounded-full bg-green-500"></div>
                        <span className="text-sm">Account active</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="h-3 w-3 rounded-full bg-muted"></div>
                        <span className="text-sm">Email verified</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="h-3 w-3 rounded-full bg-blue-500"></div>
                        <span className="text-sm">Verified user</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
          
          {/* Security Tab */}
          <TabsContent value="security">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Password</CardTitle>
                    <CardDescription>
                      Change your password to keep your account secure
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-2">
                      <Label htmlFor="currentPassword">Current Password</Label>
                      <Input id="currentPassword" type="password" />
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="newPassword">New Password</Label>
                      <Input id="newPassword" type="password" />
                      <p className="text-sm text-muted-foreground">
                        Password must be at least 8 characters and include a number and a special character.
                      </p>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="confirmPassword">Confirm New Password</Label>
                      <Input id="confirmPassword" type="password" />
                    </div>
                  </CardContent>
                  <CardFooter className="flex justify-end border-t pt-6">
                    <Button onClick={handleChangePassword}>
                      Update Password
                    </Button>
                  </CardFooter>
                </Card>
                
                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle>Login History</CardTitle>
                    <CardDescription>
                      Recent logins to your account
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {[
                        { date: "May 21, 2025", time: "11:42 AM", device: "Chrome on Windows", location: "San Francisco, CA" },
                        { date: "May 20, 2025", time: "9:15 PM", device: "Safari on iPhone", location: "San Francisco, CA" },
                        { date: "May 18, 2025", time: "3:30 PM", device: "Chrome on MacOS", location: "San Francisco, CA" }
                      ].map((login, index) => (
                        <div key={index} className="flex items-start gap-4 py-3 border-b border-muted last:border-0">
                          <div className="h-9 w-9 rounded-full bg-muted flex items-center justify-center">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                              <circle cx="12" cy="7" r="4"></circle>
                            </svg>
                          </div>
                          <div className="flex-1">
                            <div className="flex justify-between items-start">
                              <div>
                                <p className="text-sm font-medium">{login.device}</p>
                                <p className="text-xs text-muted-foreground">{login.location}</p>
                              </div>
                              <div className="text-right">
                                <p className="text-sm">{login.date}</p>
                                <p className="text-xs text-muted-foreground">{login.time}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              <div>
                <Card>
                  <CardHeader>
                    <CardTitle>Security Settings</CardTitle>
                    <CardDescription>
                      Manage account security features
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label htmlFor="twoFactorAuth">Two-Factor Authentication</Label>
                        <p className="text-sm text-muted-foreground">
                          Add an extra layer of security to your account
                        </p>
                      </div>
                      <Switch 
                        id="twoFactorAuth" 
                        checked={securitySettings.twoFactorAuth}
                        onCheckedChange={(checked) => 
                          setSecuritySettings(prev => ({ ...prev, twoFactorAuth: checked }))
                        }
                      />
                    </div>
                    
                    <Separator />
                    
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label htmlFor="loginNotifications">Login Notifications</Label>
                        <p className="text-sm text-muted-foreground">
                          Receive email alerts for new login activity
                        </p>
                      </div>
                      <Switch 
                        id="loginNotifications" 
                        checked={securitySettings.loginNotifications}
                        onCheckedChange={(checked) => 
                          setSecuritySettings(prev => ({ ...prev, loginNotifications: checked }))
                        }
                      />
                    </div>
                    
                    <Separator />
                    
                    <Button variant="outline" className="w-full">
                      Export Account Data
                    </Button>
                    
                    <Button variant="destructive" className="w-full">
                      Delete Account
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
          
          {/* Notifications Tab */}
          <TabsContent value="notifications">
            <Card>
              <CardHeader>
                <CardTitle>Notification Preferences</CardTitle>
                <CardDescription>
                  Choose how and when you want to be notified
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="emailNotifications">Email Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive notifications via email
                    </p>
                  </div>
                  <Switch 
                    id="emailNotifications" 
                    checked={notificationPrefs.emailNotifications}
                    onCheckedChange={(checked) => 
                      setNotificationPrefs(prev => ({ ...prev, emailNotifications: checked }))
                    }
                  />
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="text-md font-medium">Alerts</h3>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="analysisCompletionAlerts">Analysis Completion</Label>
                      <p className="text-sm text-muted-foreground">
                        When your video analysis is complete
                      </p>
                    </div>
                    <Switch 
                      id="analysisCompletionAlerts" 
                      checked={notificationPrefs.analysisCompletionAlerts}
                      onCheckedChange={(checked) => 
                        setNotificationPrefs(prev => ({ ...prev, analysisCompletionAlerts: checked }))
                      }
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="securityAlerts">Security Alerts</Label>
                      <p className="text-sm text-muted-foreground">
                        Important security-related notifications
                      </p>
                    </div>
                    <Switch 
                      id="securityAlerts" 
                      checked={notificationPrefs.securityAlerts}
                      onCheckedChange={(checked) => 
                        setNotificationPrefs(prev => ({ ...prev, securityAlerts: checked }))
                      }
                    />
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <h3 className="text-md font-medium">Updates & Marketing</h3>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="weeklyReportSummary">Weekly Report Summary</Label>
                      <p className="text-sm text-muted-foreground">
                        Weekly summary of your detection activities
                      </p>
                    </div>
                    <Switch 
                      id="weeklyReportSummary" 
                      checked={notificationPrefs.weeklyReportSummary}
                      onCheckedChange={(checked) => 
                        setNotificationPrefs(prev => ({ ...prev, weeklyReportSummary: checked }))
                      }
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="marketingEmails">Marketing Emails</Label>
                      <p className="text-sm text-muted-foreground">
                        News, updates, and promotional content
                      </p>
                    </div>
                    <Switch 
                      id="marketingEmails" 
                      checked={notificationPrefs.marketingEmails}
                      onCheckedChange={(checked) => 
                        setNotificationPrefs(prev => ({ ...prev, marketingEmails: checked }))
                      }
                    />
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-end border-t pt-6">
                <Button onClick={handleSaveProfile} disabled={isLoading}>
                  {isLoading ? 'Saving...' : 'Save Preferences'}
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
          
          {/* API Access Tab */}
          <TabsContent value="api">
            <Card>
              <CardHeader>
                <CardTitle>API Access</CardTitle>
                <CardDescription>
                  Manage your API keys and developer access
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="bg-muted p-4 rounded-md">
                  <h3 className="text-md font-medium mb-2">Your API Key</h3>
                  <div className="flex gap-2">
                    <Input 
                      value="••••••••••••••••••••••••••••••"
                      disabled
                      className="font-mono"
                    />
                    <Button variant="outline">
                      Show
                    </Button>
                    <Button variant="outline">
                      Copy
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground mt-2">
                    Never share your API key. You can regenerate it if it gets compromised.
                  </p>
                </div>
                
                <div className="flex justify-between">
                  <Button variant="outline">
                    Regenerate Key
                  </Button>
                  <Button variant="destructive">
                    Revoke Access
                  </Button>
                </div>
                
                <Separator />
                
                <div>
                  <h3 className="text-md font-medium mb-4">API Usage Stats</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Monthly Quota</span>
                        <span className="text-sm text-muted-foreground">346 / 500 requests</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2.5">
                        <div className="bg-primary h-2.5 rounded-full" style={{ width: '69%' }}></div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-muted/50 p-4 rounded-md text-center">
                        <p className="text-2xl font-semibold">346</p>
                        <p className="text-sm text-muted-foreground">Requests This Month</p>
                      </div>
                      <div className="bg-muted/50 p-4 rounded-md text-center">
                        <p className="text-2xl font-semibold">1.2s</p>
                        <p className="text-sm text-muted-foreground">Average Response Time</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <Separator />
                
                <div>
                  <h3 className="text-md font-medium mb-4">Documentation</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Access our comprehensive API documentation to integrate with our deepfake detection services.
                  </p>
                  <Button variant="outline">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                      <polyline points="14 2 14 8 20 8"/>
                      <line x1="16" y1="13" x2="8" y2="13"/>
                      <line x1="16" y1="17" x2="8" y2="17"/>
                      <polyline points="10 9 9 9 8 9"/>
                    </svg>
                    View API Documentation
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}