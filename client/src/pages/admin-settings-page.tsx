import { useState, useEffect } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import Sidebar from "@/components/layout/sidebar";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";

export default function AdminSettingsPage() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isCacheClearing, setIsCacheClearing] = useState(false);
  
  // Mock settings - would be replaced with actual API data
  const [generalSettings, setGeneralSettings] = useState({
    siteName: "DeepFake Detector",
    maxFileSize: 100,
    allowedFileTypes: ["mp4", "mov", "avi"],
    defaultLanguage: "en",
    systemTheme: "dark",
    enableUserRegistration: true,
    requireEmailVerification: true,
    maintenanceMode: false
  });
  
  const [apiSettings, setApiSettings] = useState({
    geminiApiEnabled: true,
    apiThrottling: true,
    requestsPerMinute: 60,
    maxConcurrentRequests: 10,
    apiTimeout: 30
  });
  
  const [notificationSettings, setNotificationSettings] = useState({
    emailNotifications: true,
    adminAlerts: true,
    systemUpdates: true,
    securityAlerts: true,
    userReports: true
  });
  
  const [securitySettings, setSecuritySettings] = useState({
    twoFactorAuth: false,
    passwordStrength: "medium",
    sessionTimeout: 60,
    ipWhitelist: [] as string[],
    loginAttempts: 5
  });

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

  const handleSaveSettings = () => {
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      toast({
        title: "Settings Saved",
        description: "Your system settings have been updated successfully.",
      });
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar isAdmin />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Admin Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">System Settings</h1>
            <p className="text-muted-foreground">Configure platform settings and preferences</p>
          </div>
          
          <Button onClick={handleSaveSettings} disabled={isLoading}>
            {isLoading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Saving...
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                  <polyline points="17 21 17 13 7 13 7 21"/>
                  <polyline points="7 3 7 8 15 8"/>
                </svg>
                Save Changes
              </>
            )}
          </Button>
        </div>

        {/* Settings Tabs */}
        <Tabs defaultValue="general" className="w-full">
          <TabsList className="mb-6 w-full md:w-auto">
            <TabsTrigger value="general">General</TabsTrigger>
            <TabsTrigger value="api">API & Performance</TabsTrigger>
            <TabsTrigger value="notifications">Notifications</TabsTrigger>
            <TabsTrigger value="security">Security</TabsTrigger>
            <TabsTrigger value="cache">Cache Management</TabsTrigger>
          </TabsList>
          
          {/* General Settings Tab */}
          <TabsContent value="general">
            <Card>
              <CardHeader>
                <CardTitle>General Settings</CardTitle>
                <CardDescription>
                  Manage basic system configuration and appearance
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="siteName">Site Name</Label>
                    <Input 
                      id="siteName" 
                      value={generalSettings.siteName}
                      onChange={(e) => setGeneralSettings({...generalSettings, siteName: e.target.value})}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="maxFileSize">Maximum File Size (MB)</Label>
                    <Input 
                      id="maxFileSize" 
                      type="number"
                      value={generalSettings.maxFileSize}
                      onChange={(e) => setGeneralSettings({...generalSettings, maxFileSize: parseInt(e.target.value)})}
                    />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="allowedFileTypes">Allowed File Types</Label>
                  <div className="flex flex-wrap gap-2">
                    {generalSettings.allowedFileTypes.map((type, index) => (
                      <div key={index} className="flex items-center bg-muted rounded-md px-3 py-1">
                        <span>{type}</span>
                        <button 
                          className="ml-2 text-muted-foreground hover:text-destructive"
                          onClick={() => {
                            const newTypes = [...generalSettings.allowedFileTypes];
                            newTypes.splice(index, 1);
                            setGeneralSettings({...generalSettings, allowedFileTypes: newTypes});
                          }}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => {
                        const newType = prompt("Enter file extension (without dot)");
                        if (newType && !generalSettings.allowedFileTypes.includes(newType)) {
                          setGeneralSettings({
                            ...generalSettings, 
                            allowedFileTypes: [...generalSettings.allowedFileTypes, newType]
                          });
                        }
                      }}
                    >
                      Add Type
                    </Button>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="defaultLanguage">Default Language</Label>
                    <Select 
                      value={generalSettings.defaultLanguage}
                      onValueChange={(value) => setGeneralSettings({...generalSettings, defaultLanguage: value})}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select language" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="en">English</SelectItem>
                        <SelectItem value="es">Spanish</SelectItem>
                        <SelectItem value="fr">French</SelectItem>
                        <SelectItem value="de">German</SelectItem>
                        <SelectItem value="zh">Chinese</SelectItem>
                        <SelectItem value="ja">Japanese</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="systemTheme">System Theme</Label>
                    <Select 
                      value={generalSettings.systemTheme}
                      onValueChange={(value) => setGeneralSettings({...generalSettings, systemTheme: value})}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select theme" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="light">Light</SelectItem>
                        <SelectItem value="dark">Dark</SelectItem>
                        <SelectItem value="system">System Default</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <Separator className="my-4" />
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="enableRegistration">Enable User Registration</Label>
                      <p className="text-sm text-muted-foreground">Allow new users to register accounts</p>
                    </div>
                    <Switch 
                      id="enableRegistration" 
                      checked={generalSettings.enableUserRegistration}
                      onCheckedChange={(value) => setGeneralSettings({...generalSettings, enableUserRegistration: value})}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="requireEmailVerification">Require Email Verification</Label>
                      <p className="text-sm text-muted-foreground">Users must verify email before accessing features</p>
                    </div>
                    <Switch 
                      id="requireEmailVerification" 
                      checked={generalSettings.requireEmailVerification}
                      onCheckedChange={(value) => setGeneralSettings({...generalSettings, requireEmailVerification: value})}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="maintenanceMode" className="text-destructive font-medium">Maintenance Mode</Label>
                      <p className="text-sm text-muted-foreground">Take the site offline for maintenance</p>
                    </div>
                    <Switch 
                      id="maintenanceMode" 
                      checked={generalSettings.maintenanceMode}
                      onCheckedChange={(value) => setGeneralSettings({...generalSettings, maintenanceMode: value})}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* API Settings Tab */}
          <TabsContent value="api">
            <Card>
              <CardHeader>
                <CardTitle>API & Performance Settings</CardTitle>
                <CardDescription>
                  Configure API connections and system performance parameters
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="geminiApiEnabled">Google Gemini API</Label>
                    <p className="text-sm text-muted-foreground">Enable connection to Google Gemini API</p>
                  </div>
                  <Switch 
                    id="geminiApiEnabled" 
                    checked={apiSettings.geminiApiEnabled}
                    onCheckedChange={(value) => setApiSettings({...apiSettings, geminiApiEnabled: value})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="apiThrottling">API Request Throttling</Label>
                    <p className="text-sm text-muted-foreground">Limit rate of API requests to prevent overload</p>
                  </div>
                  <Switch 
                    id="apiThrottling" 
                    checked={apiSettings.apiThrottling}
                    onCheckedChange={(value) => setApiSettings({...apiSettings, apiThrottling: value})}
                  />
                </div>
                
                <Separator className="my-4" />
                
                <div className="space-y-8">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="requestsPerMinute">Requests Per Minute: {apiSettings.requestsPerMinute}</Label>
                      <span className="text-sm text-muted-foreground w-12 text-right">{apiSettings.requestsPerMinute}</span>
                    </div>
                    <Slider
                      id="requestsPerMinute"
                      min={10}
                      max={200}
                      step={5}
                      value={[apiSettings.requestsPerMinute]}
                      onValueChange={(value) => setApiSettings({...apiSettings, requestsPerMinute: value[0]})}
                    />
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="maxConcurrentRequests">Max Concurrent Requests: {apiSettings.maxConcurrentRequests}</Label>
                      <span className="text-sm text-muted-foreground w-12 text-right">{apiSettings.maxConcurrentRequests}</span>
                    </div>
                    <Slider
                      id="maxConcurrentRequests"
                      min={1}
                      max={50}
                      step={1}
                      value={[apiSettings.maxConcurrentRequests]}
                      onValueChange={(value) => setApiSettings({...apiSettings, maxConcurrentRequests: value[0]})}
                    />
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="apiTimeout">API Timeout (seconds): {apiSettings.apiTimeout}</Label>
                      <span className="text-sm text-muted-foreground w-12 text-right">{apiSettings.apiTimeout}</span>
                    </div>
                    <Slider
                      id="apiTimeout"
                      min={5}
                      max={120}
                      step={5}
                      value={[apiSettings.apiTimeout]}
                      onValueChange={(value) => setApiSettings({...apiSettings, apiTimeout: value[0]})}
                    />
                  </div>
                </div>
                
                <div className="pt-4">
                  <Button variant="outline">
                    Test API Connection
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Notification Settings Tab */}
          <TabsContent value="notifications">
            <Card>
              <CardHeader>
                <CardTitle>Notification Settings</CardTitle>
                <CardDescription>
                  Configure system notifications and alerts
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="emailNotifications">Email Notifications</Label>
                      <p className="text-sm text-muted-foreground">Send important notifications via email</p>
                    </div>
                    <Switch 
                      id="emailNotifications" 
                      checked={notificationSettings.emailNotifications}
                      onCheckedChange={(value) => setNotificationSettings({...notificationSettings, emailNotifications: value})}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="adminAlerts">Admin Alerts</Label>
                      <p className="text-sm text-muted-foreground">Notify administrators about critical events</p>
                    </div>
                    <Switch 
                      id="adminAlerts" 
                      checked={notificationSettings.adminAlerts}
                      onCheckedChange={(value) => setNotificationSettings({...notificationSettings, adminAlerts: value})}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="systemUpdates">System Update Notifications</Label>
                      <p className="text-sm text-muted-foreground">Notify about available system updates</p>
                    </div>
                    <Switch 
                      id="systemUpdates" 
                      checked={notificationSettings.systemUpdates}
                      onCheckedChange={(value) => setNotificationSettings({...notificationSettings, systemUpdates: value})}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="securityAlerts">Security Alerts</Label>
                      <p className="text-sm text-muted-foreground">Notify about security events and issues</p>
                    </div>
                    <Switch 
                      id="securityAlerts" 
                      checked={notificationSettings.securityAlerts}
                      onCheckedChange={(value) => setNotificationSettings({...notificationSettings, securityAlerts: value})}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="userReports">User Reports</Label>
                      <p className="text-sm text-muted-foreground">Notify about user-submitted reports</p>
                    </div>
                    <Switch 
                      id="userReports" 
                      checked={notificationSettings.userReports}
                      onCheckedChange={(value) => setNotificationSettings({...notificationSettings, userReports: value})}
                    />
                  </div>
                </div>
                
                <Separator className="my-4" />
                
                <div className="space-y-2">
                  <Label htmlFor="emailServer">Email Server Settings</Label>
                  <Card className="border border-muted">
                    <CardContent className="pt-6 space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="smtpServer">SMTP Server</Label>
                          <Input id="smtpServer" placeholder="smtp.example.com" />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="smtpPort">SMTP Port</Label>
                          <Input id="smtpPort" placeholder="587" />
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="smtpUsername">SMTP Username</Label>
                          <Input id="smtpUsername" placeholder="user@example.com" />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="smtpPassword">SMTP Password</Label>
                          <Input id="smtpPassword" type="password" placeholder="••••••••" />
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label htmlFor="useTLS">Use TLS/SSL</Label>
                        </div>
                        <Switch id="useTLS" defaultChecked={true} />
                      </div>
                    </CardContent>
                    <CardFooter className="flex justify-between border-t bg-muted/50 px-6 py-3">
                      <Button variant="outline" size="sm">
                        Test Connection
                      </Button>
                      <Button size="sm">
                        Save Email Settings
                      </Button>
                    </CardFooter>
                  </Card>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Security Settings Tab */}
          <TabsContent value="security">
            <Card>
              <CardHeader>
                <CardTitle>Security Settings</CardTitle>
                <CardDescription>
                  Configure security policies and access controls
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="twoFactorAuth">Two-Factor Authentication</Label>
                    <p className="text-sm text-muted-foreground">Require 2FA for all admin accounts</p>
                  </div>
                  <Switch 
                    id="twoFactorAuth" 
                    checked={securitySettings.twoFactorAuth}
                    onCheckedChange={(value) => setSecuritySettings({...securitySettings, twoFactorAuth: value})}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="passwordStrength">Password Strength Requirement</Label>
                  <Select 
                    value={securitySettings.passwordStrength}
                    onValueChange={(value) => setSecuritySettings({...securitySettings, passwordStrength: value})}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select requirement" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low - Minimum 6 characters</SelectItem>
                      <SelectItem value="medium">Medium - 8+ chars with numbers</SelectItem>
                      <SelectItem value="high">High - 10+ chars with numbers & symbols</SelectItem>
                      <SelectItem value="extreme">Extreme - 12+ with upper, lower, numbers & symbols</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="sessionTimeout">Session Timeout (minutes): {securitySettings.sessionTimeout}</Label>
                    <span className="text-sm text-muted-foreground w-12 text-right">{securitySettings.sessionTimeout}</span>
                  </div>
                  <Slider
                    id="sessionTimeout"
                    min={5}
                    max={240}
                    step={5}
                    value={[securitySettings.sessionTimeout]}
                    onValueChange={(value) => setSecuritySettings({...securitySettings, sessionTimeout: value[0]})}
                  />
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="loginAttempts">Max Login Attempts: {securitySettings.loginAttempts}</Label>
                    <span className="text-sm text-muted-foreground w-12 text-right">{securitySettings.loginAttempts}</span>
                  </div>
                  <Slider
                    id="loginAttempts"
                    min={1}
                    max={10}
                    step={1}
                    value={[securitySettings.loginAttempts]}
                    onValueChange={(value) => setSecuritySettings({...securitySettings, loginAttempts: value[0]})}
                  />
                </div>
                
                <Separator className="my-4" />
                
                <div className="space-y-2">
                  <Label htmlFor="ipWhitelist">IP Whitelist</Label>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {securitySettings.ipWhitelist.length > 0 ? (
                      securitySettings.ipWhitelist.map((ip, index) => (
                        <div key={index} className="flex items-center bg-muted rounded-md px-3 py-1">
                          <span>{ip}</span>
                          <button 
                            className="ml-2 text-muted-foreground hover:text-destructive"
                            onClick={() => {
                              const newList = [...securitySettings.ipWhitelist];
                              newList.splice(index, 1);
                              setSecuritySettings({...securitySettings, ipWhitelist: newList});
                            }}
                          >
                            ×
                          </button>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground">No IP addresses whitelisted</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Input id="newIp" placeholder="Enter IP address" />
                    <Button 
                      variant="outline"
                      onClick={() => {
                        // Just for demo - would validate IP in real implementation
                        const input = document.getElementById("newIp") as HTMLInputElement;
                        const newIp = input.value.trim();
                        if (newIp) {
                          const updatedList = [...securitySettings.ipWhitelist];
                          if (!updatedList.includes(newIp)) {
                            updatedList.push(newIp);
                          }
                          setSecuritySettings({
                            ...securitySettings,
                            ipWhitelist: updatedList
                          });
                          input.value = "";
                        }
                      }}
                    >
                      Add IP
                    </Button>
                  </div>
                </div>
                
                <div className="pt-4 flex gap-4">
                  <Button variant="outline" className="text-destructive border-destructive hover:bg-destructive/10">
                    Reset All Security Settings
                  </Button>
                  <Button variant="default">
                    Run Security Scan
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Cache Management Tab */}
          <TabsContent value="cache">
            <Card>
              <CardHeader>
                <CardTitle>Cache Management</CardTitle>
                <CardDescription>
                  Control system cache and optimize memory usage
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Cache Info Section */}
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Video Cache Statistics</h3>
                  
                  {/* Video Cache Statistics */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="bg-muted/50">
                      <CardContent className="pt-6">
                        <div className="text-center">
                          <div className="text-2xl font-bold">{user?.videoCount || 0}</div>
                          <p className="text-sm text-muted-foreground">Videos Cached</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="bg-muted/50">
                      <CardContent className="pt-6">
                        <div className="text-center">
                          <div className="text-2xl font-bold">{(user?.videoCount || 0) * 15} MB</div>
                          <p className="text-sm text-muted-foreground">Estimated Cache Size</p>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="bg-muted/50">
                      <CardContent className="pt-6">
                        <div className="text-center">
                          <div className="text-2xl font-bold">Never</div>
                          <p className="text-sm text-muted-foreground">Last Cache Clear</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
                
                <Separator className="my-4" />
                
                {/* Cache Management Controls */}
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Cache Management Controls</h3>
                  
                  <div className="p-4 border border-destructive/20 rounded-md bg-destructive/5">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div className="space-y-1">
                        <h4 className="font-medium text-destructive">Clear Video Cache</h4>
                        <p className="text-sm text-muted-foreground">
                          This will remove all cached video files and analysis results from memory.
                          Use this when the system is using too much memory or you want to free up space.
                        </p>
                      </div>
                      
                      <Button 
                        variant="destructive" 
                        onClick={() => {
                          if (confirm("Are you sure you want to clear the video cache? This action cannot be undone.")) {
                            setIsCacheClearing(true);
                            
                            // Call the API to clear cache
                            fetch('/api/admin/clear-cache', {
                              method: 'POST',
                              headers: {
                                'Content-Type': 'application/json',
                              },
                            })
                            .then(response => response.json())
                            .then(data => {
                              setIsCacheClearing(false);
                              if (data.success) {
                                toast({
                                  title: "Cache Cleared",
                                  description: data.message || "Video cache has been successfully cleared.",
                                });
                              } else {
                                throw new Error(data.error || "Failed to clear cache");
                              }
                            })
                            .catch(error => {
                              setIsCacheClearing(false);
                              toast({
                                title: "Error",
                                description: error.message || "Failed to clear video cache.",
                                variant: "destructive",
                              });
                            });
                          }
                        }}
                        disabled={isCacheClearing}
                      >
                        {isCacheClearing ? (
                          <>
                            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Clearing Cache...
                          </>
                        ) : (
                          <>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                              <path d="M3 6h18"></path>
                              <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
                              <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
                              <line x1="10" y1="11" x2="10" y2="17"></line>
                              <line x1="14" y1="11" x2="14" y2="17"></line>
                            </svg>
                            Clear Video Cache
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                  
                  <div className="p-4 border rounded-md bg-muted/50">
                    <div className="space-y-2">
                      <h4 className="font-medium">Automatic Cache Management</h4>
                      <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                          <Label htmlFor="autoCacheClean">Enable Automatic Cache Cleanup</Label>
                          <p className="text-sm text-muted-foreground">System will automatically clear old cache entries</p>
                        </div>
                        <Switch id="autoCacheClean" />
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}