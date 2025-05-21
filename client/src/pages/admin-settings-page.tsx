import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";

export default function AdminSettingsPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  
  const [systemSettings, setSystemSettings] = useState({
    maxVideoSize: 100,
    maxVideoDuration: 5,
    dailyUploadLimit: 10,
    concurrentAnalysis: 5,
    analysisTimeout: 15,
    enableModelTraining: true,
    autoDeleteAnalysisAfterDays: 30,
    maintenanceMode: false,
    apiTimeout: 60,
    userRegistrationType: "open",
    securityLevel: "high",
    adminApprovalRequired: false,
    debugLogging: false,
    storageType: "local"
  });
  
  const handleSettingChange = (setting: string, value: any) => {
    setSystemSettings(prev => ({
      ...prev,
      [setting]: value
    }));
    
    toast({
      title: "Setting updated",
      description: "The system setting has been updated successfully.",
    });
  };

  const handleSaveSettings = () => {
    toast({
      title: "Settings saved",
      description: "All system settings have been saved successfully.",
    });
  };
  
  const updateSystem = () => {
    // Simulate a system update
    toast({
      title: "System update initiated",
      description: "The system update process has begun. This may take a few minutes.",
    });
  };

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isAdmin={true} />
      
      <div className="flex-1 ml-20 md:ml-64 p-8">
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
            <h1 className="text-3xl font-bold">System Settings</h1>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <Button variant="outline" onClick={updateSystem}>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><rect width="20" height="20" x="2" y="2" rx="5" ry="5"/><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/><line x1="17.5" x2="17.51" y1="6.5" y2="6.5"/></svg>
                Check for Updates
              </Button>
              
              <Button onClick={handleSaveSettings}>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
                Save All Settings
              </Button>
            </div>
          </div>
          
          <Tabs defaultValue="general" className="w-full">
            <TabsList className="grid w-full md:w-auto md:inline-flex grid-cols-3 md:grid-cols-none mb-8">
              <TabsTrigger value="general">General</TabsTrigger>
              <TabsTrigger value="security">Security & Privacy</TabsTrigger>
              <TabsTrigger value="advanced">Advanced</TabsTrigger>
            </TabsList>
            
            <TabsContent value="general">
              <div className="grid gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Upload Settings</CardTitle>
                    <CardDescription>
                      Configure video upload and analysis parameters
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="maxVideoSize">Maximum Video Size (MB)</Label>
                        <p className="text-sm text-muted-foreground">
                          Set the maximum allowed file size for uploads
                        </p>
                      </div>
                      <Input
                        id="maxVideoSize"
                        type="number"
                        value={systemSettings.maxVideoSize}
                        onChange={(e) => handleSettingChange("maxVideoSize", parseInt(e.target.value))}
                        className="w-full md:w-[150px]"
                      />
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="maxVideoDuration">Maximum Video Duration (min)</Label>
                        <p className="text-sm text-muted-foreground">
                          Maximum length of videos that can be analyzed
                        </p>
                      </div>
                      <Input
                        id="maxVideoDuration"
                        type="number"
                        value={systemSettings.maxVideoDuration}
                        onChange={(e) => handleSettingChange("maxVideoDuration", parseInt(e.target.value))}
                        className="w-full md:w-[150px]"
                      />
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="dailyUploadLimit">Daily Upload Limit (per user)</Label>
                        <p className="text-sm text-muted-foreground">
                          Number of videos a user can upload daily
                        </p>
                      </div>
                      <Input
                        id="dailyUploadLimit"
                        type="number"
                        value={systemSettings.dailyUploadLimit}
                        onChange={(e) => handleSettingChange("dailyUploadLimit", parseInt(e.target.value))}
                        className="w-full md:w-[150px]"
                      />
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Analysis Engine</CardTitle>
                    <CardDescription>
                      Configure deepfake detection settings
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="concurrentAnalysis">Concurrent Analyses</Label>
                        <p className="text-sm text-muted-foreground">
                          Maximum number of simultaneous analyses
                        </p>
                      </div>
                      <Input
                        id="concurrentAnalysis"
                        type="number"
                        value={systemSettings.concurrentAnalysis}
                        onChange={(e) => handleSettingChange("concurrentAnalysis", parseInt(e.target.value))}
                        className="w-full md:w-[150px]"
                      />
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="analysisTimeout">Analysis Timeout (min)</Label>
                        <p className="text-sm text-muted-foreground">
                          Maximum time before an analysis times out
                        </p>
                      </div>
                      <Input
                        id="analysisTimeout"
                        type="number"
                        value={systemSettings.analysisTimeout}
                        onChange={(e) => handleSettingChange("analysisTimeout", parseInt(e.target.value))}
                        className="w-full md:w-[150px]"
                      />
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <p className="font-medium">Enable Model Training</p>
                        <p className="text-sm text-muted-foreground">
                          Allow system to use uploaded videos for model improvement
                        </p>
                      </div>
                      <Switch 
                        checked={systemSettings.enableModelTraining} 
                        onCheckedChange={(checked) => handleSettingChange("enableModelTraining", checked)}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            <TabsContent value="security">
              <div className="grid gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>User Authentication</CardTitle>
                    <CardDescription>
                      Configure user access and registration settings
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <p className="font-medium">User Registration</p>
                        <p className="text-sm text-muted-foreground">
                          Control how new users can register on the platform
                        </p>
                      </div>
                      <Select 
                        value={systemSettings.userRegistrationType} 
                        onValueChange={(value) => handleSettingChange("userRegistrationType", value)}
                      >
                        <SelectTrigger className="w-full md:w-[200px]">
                          <SelectValue placeholder="Registration type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="open">Open Registration</SelectItem>
                          <SelectItem value="invite">Invite Only</SelectItem>
                          <SelectItem value="closed">Closed</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <p className="font-medium">Security Level</p>
                        <p className="text-sm text-muted-foreground">
                          Set password and account security requirements
                        </p>
                      </div>
                      <Select 
                        value={systemSettings.securityLevel} 
                        onValueChange={(value) => handleSettingChange("securityLevel", value)}
                      >
                        <SelectTrigger className="w-full md:w-[200px]">
                          <SelectValue placeholder="Security level" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="standard">Standard</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                          <SelectItem value="maximum">Maximum</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <p className="font-medium">Admin Approval</p>
                        <p className="text-sm text-muted-foreground">
                          Require admin approval for new account registration
                        </p>
                      </div>
                      <Switch 
                        checked={systemSettings.adminApprovalRequired} 
                        onCheckedChange={(checked) => handleSettingChange("adminApprovalRequired", checked)}
                      />
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Privacy Settings</CardTitle>
                    <CardDescription>
                      Configure data retention and privacy options
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="autoDeleteAnalysis">Auto-Delete Analysis Data (days)</Label>
                        <p className="text-sm text-muted-foreground">
                          Automatically remove analysis data after this period
                        </p>
                      </div>
                      <Input
                        id="autoDeleteAnalysis"
                        type="number"
                        value={systemSettings.autoDeleteAnalysisAfterDays}
                        onChange={(e) => handleSettingChange("autoDeleteAnalysisAfterDays", parseInt(e.target.value))}
                        className="w-full md:w-[150px]"
                      />
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <p className="font-medium">Video Storage Location</p>
                        <p className="text-sm text-muted-foreground">
                          Select where uploaded videos are stored
                        </p>
                      </div>
                      <Select 
                        value={systemSettings.storageType} 
                        onValueChange={(value) => handleSettingChange("storageType", value)}
                      >
                        <SelectTrigger className="w-full md:w-[200px]">
                          <SelectValue placeholder="Storage type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="local">Local Storage</SelectItem>
                          <SelectItem value="cloud">Cloud Storage</SelectItem>
                          <SelectItem value="distributed">Distributed Storage</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            <TabsContent value="advanced">
              <div className="grid gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>System Maintenance</CardTitle>
                    <CardDescription>
                      Advanced system configuration options
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <p className="font-medium">Maintenance Mode</p>
                        <p className="text-sm text-muted-foreground">
                          Put the system in maintenance mode (users cannot access)
                        </p>
                      </div>
                      <Switch 
                        checked={systemSettings.maintenanceMode} 
                        onCheckedChange={(checked) => handleSettingChange("maintenanceMode", checked)}
                      />
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="apiTimeout">API Timeout (seconds)</Label>
                        <p className="text-sm text-muted-foreground">
                          Maximum time for API requests to complete
                        </p>
                      </div>
                      <Input
                        id="apiTimeout"
                        type="number"
                        value={systemSettings.apiTimeout}
                        onChange={(e) => handleSettingChange("apiTimeout", parseInt(e.target.value))}
                        className="w-full md:w-[150px]"
                      />
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <p className="font-medium">Debug Logging</p>
                        <p className="text-sm text-muted-foreground">
                          Enable verbose logging for system debugging
                        </p>
                      </div>
                      <Switch 
                        checked={systemSettings.debugLogging} 
                        onCheckedChange={(checked) => handleSettingChange("debugLogging", checked)}
                      />
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Gemini API Configuration</CardTitle>
                    <CardDescription>
                      Configure settings for the Google Gemini API integration
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="apiKey">Gemini API Key</Label>
                        <p className="text-sm text-muted-foreground">
                          API key for connecting to Google's Gemini AI
                        </p>
                      </div>
                      <div className="flex items-center gap-2 w-full md:w-auto">
                        <Input
                          id="apiKey"
                          type="password"
                          value="••••••••••••••••••••••••••••••"
                          className="w-full md:w-[250px]"
                          readOnly
                        />
                        <Button variant="outline" size="sm">Update</Button>
                      </div>
                    </div>
                    
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div>
                        <Label htmlFor="model">AI Model</Label>
                        <p className="text-sm text-muted-foreground">
                          Select the Gemini model version to use
                        </p>
                      </div>
                      <Select defaultValue="gemini-pro">
                        <SelectTrigger className="w-full md:w-[200px]">
                          <SelectValue placeholder="Select model" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                          <SelectItem value="gemini-ultra">Gemini Ultra</SelectItem>
                          <SelectItem value="gemini-pro-vision">Gemini Pro Vision</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                  <CardFooter className="border-t px-6 py-4">
                    <div className="flex justify-between items-center w-full">
                      <div className="flex items-center text-sm text-muted-foreground">
                        <span className="inline-block w-2 h-2 rounded-full bg-green-500 mr-2"></span>
                        API Connected
                      </div>
                      <Button variant="outline" size="sm">
                        Test Connection
                      </Button>
                    </div>
                  </CardFooter>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>System Operations</CardTitle>
                    <CardDescription>
                      Advanced system maintenance operations
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <Button variant="outline">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><path d="M12 18v-6"/><path d="m9 15 3 3 3-3"/></svg>
                        Backup Database
                      </Button>
                      
                      <Button variant="outline">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><path d="M12 12v6"/><path d="m15 15-3 3-3-3"/></svg>
                        Restore Backup
                      </Button>
                      
                      <Button variant="outline">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg>
                        Clear Cache
                      </Button>
                      
                      <Button variant="outline" className="text-destructive hover:text-destructive hover:bg-destructive/10">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
                        Reset System
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}