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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";

export default function SettingsPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  
  // App settings
  const [appSettings, setAppSettings] = useState({
    theme: "dark",
    language: "en",
    autoPlayVideos: true,
    highContrastMode: false,
    compactMode: false,
    reducedMotion: false,
    autoSave: true,
    saveInterval: 5, // minutes
  });

  // Privacy settings
  const [privacySettings, setPrivacySettings] = useState({
    shareAnalytics: true,
    storeHistory: true,
    publicProfile: false,
    dataRetention: "90days",
    cookiePreference: "essential"
  });

  // Display settings
  const [displaySettings, setDisplaySettings] = useState({
    fontSize: 16,
    lineHeight: 1.5,
    colorScheme: "default"
  });

  const handleSaveSettings = () => {
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      toast({
        title: "Settings saved",
        description: "Your preferences have been updated successfully."
      });
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      
      <div className="ml-20 md:ml-64 p-6 pt-8 min-h-screen">
        {/* Page Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold">Settings</h1>
            <p className="text-muted-foreground">Customize your application experience</p>
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
              'Save All Settings'
            )}
          </Button>
        </div>

        <Tabs defaultValue="appearance" className="w-full">
          <TabsList className="mb-6 w-full md:w-auto">
            <TabsTrigger value="appearance">Appearance</TabsTrigger>
            <TabsTrigger value="privacy">Privacy</TabsTrigger>
            <TabsTrigger value="accessibility">Accessibility</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>
          
          {/* Appearance Tab */}
          <TabsContent value="appearance">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Theme Settings</CardTitle>
                  <CardDescription>
                    Customize the look and feel of the application
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="theme">Color Theme</Label>
                    <RadioGroup 
                      value={appSettings.theme} 
                      onValueChange={(value) => setAppSettings({...appSettings, theme: value})} 
                      className="flex flex-col space-y-1"
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="light" id="theme-light" />
                        <Label htmlFor="theme-light">Light</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="dark" id="theme-dark" />
                        <Label htmlFor="theme-dark">Dark</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="system" id="theme-system" />
                        <Label htmlFor="theme-system">System Preference</Label>
                      </div>
                    </RadioGroup>
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-2">
                    <Label htmlFor="colorScheme">Color Scheme</Label>
                    <Select 
                      value={displaySettings.colorScheme}
                      onValueChange={(value) => setDisplaySettings({...displaySettings, colorScheme: value})}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select color scheme" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="default">Default</SelectItem>
                        <SelectItem value="blue">Blue</SelectItem>
                        <SelectItem value="purple">Purple</SelectItem>
                        <SelectItem value="green">Green</SelectItem>
                        <SelectItem value="orange">Orange</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <Label htmlFor="fontSize">Font Size: {displaySettings.fontSize}px</Label>
                        <span className="text-sm text-muted-foreground w-12 text-right">{displaySettings.fontSize}px</span>
                      </div>
                      <Slider
                        id="fontSize"
                        min={12}
                        max={24}
                        step={1}
                        value={[displaySettings.fontSize]}
                        onValueChange={(value) => setDisplaySettings({...displaySettings, fontSize: value[0]})}
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <Label htmlFor="lineHeight">Line Height: {displaySettings.lineHeight}</Label>
                        <span className="text-sm text-muted-foreground w-12 text-right">{displaySettings.lineHeight}</span>
                      </div>
                      <Slider
                        id="lineHeight"
                        min={1}
                        max={2}
                        step={0.1}
                        value={[displaySettings.lineHeight]}
                        onValueChange={(value) => setDisplaySettings({...displaySettings, lineHeight: value[0]})}
                      />
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="compactMode">Compact Mode</Label>
                      <p className="text-sm text-muted-foreground">
                        Reduce spacing for a more compact interface
                      </p>
                    </div>
                    <Switch 
                      id="compactMode" 
                      checked={appSettings.compactMode}
                      onCheckedChange={(checked) => setAppSettings({...appSettings, compactMode: checked})}
                    />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Display Settings</CardTitle>
                  <CardDescription>
                    Configure how content is displayed
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="language">Language</Label>
                    <Select 
                      value={appSettings.language}
                      onValueChange={(value) => setAppSettings({...appSettings, language: value})}
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
                  
                  <Separator />
                  
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label htmlFor="autoPlayVideos">Auto-Play Videos</Label>
                        <p className="text-sm text-muted-foreground">
                          Automatically play videos when viewing analyses
                        </p>
                      </div>
                      <Switch 
                        id="autoPlayVideos" 
                        checked={appSettings.autoPlayVideos}
                        onCheckedChange={(checked) => setAppSettings({...appSettings, autoPlayVideos: checked})}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label htmlFor="autoSave">Auto-Save</Label>
                        <p className="text-sm text-muted-foreground">
                          Automatically save your work periodically
                        </p>
                      </div>
                      <Switch 
                        id="autoSave" 
                        checked={appSettings.autoSave}
                        onCheckedChange={(checked) => setAppSettings({...appSettings, autoSave: checked})}
                      />
                    </div>
                    
                    {appSettings.autoSave && (
                      <div className="space-y-2 pl-6 border-l-2 border-muted ml-2">
                        <div className="flex justify-between items-center">
                          <Label htmlFor="saveInterval">Save Interval: {appSettings.saveInterval} minutes</Label>
                          <span className="text-sm text-muted-foreground w-12 text-right">{appSettings.saveInterval}m</span>
                        </div>
                        <Slider
                          id="saveInterval"
                          min={1}
                          max={15}
                          step={1}
                          value={[appSettings.saveInterval]}
                          onValueChange={(value) => setAppSettings({...appSettings, saveInterval: value[0]})}
                        />
                      </div>
                    )}
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-6">
                    <Button variant="outline" className="w-full">
                      Reset to Default Theme
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>Theme Preview</CardTitle>
                  <CardDescription>
                    Preview your selected theme settings
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className={`p-6 rounded-md ${appSettings.theme === 'dark' ? 'bg-zinc-900' : 'bg-white'} border transition-all`}>
                    <div className="space-y-6">
                      <div>
                        <h3 className={`text-xl font-semibold ${appSettings.theme === 'dark' ? 'text-white' : 'text-black'}`}>
                          Theme Preview
                        </h3>
                        <p className={`${appSettings.theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`} style={{ fontSize: `${displaySettings.fontSize}px`, lineHeight: displaySettings.lineHeight }}>
                          This is how your text will appear with the current settings.
                        </p>
                      </div>
                      
                      <div className="flex space-x-2">
                        <Button variant="default" className="bg-primary hover:bg-primary/90">
                          Primary Button
                        </Button>
                        <Button variant="secondary">
                          Secondary Button
                        </Button>
                        <Button variant="outline">
                          Outline Button
                        </Button>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4">
                        <div className={`p-4 rounded-md ${appSettings.theme === 'dark' ? 'bg-zinc-800' : 'bg-gray-100'}`}>
                          <p className="text-sm font-medium">Card Element</p>
                        </div>
                        <div className={`p-4 rounded-md ${appSettings.theme === 'dark' ? 'bg-zinc-800' : 'bg-gray-100'}`}>
                          <p className="text-sm font-medium">Card Element</p>
                        </div>
                        <div className={`p-4 rounded-md ${appSettings.theme === 'dark' ? 'bg-zinc-800' : 'bg-gray-100'}`}>
                          <p className="text-sm font-medium">Card Element</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          {/* Privacy Tab */}
          <TabsContent value="privacy">
            <Card>
              <CardHeader>
                <CardTitle>Privacy Settings</CardTitle>
                <CardDescription>
                  Control how your data is used and stored
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="shareAnalytics">Usage Analytics</Label>
                    <p className="text-sm text-muted-foreground">
                      Share anonymous usage data to help improve the service
                    </p>
                  </div>
                  <Switch 
                    id="shareAnalytics" 
                    checked={privacySettings.shareAnalytics}
                    onCheckedChange={(checked) => setPrivacySettings({...privacySettings, shareAnalytics: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="storeHistory">Analysis History</Label>
                    <p className="text-sm text-muted-foreground">
                      Store your video analysis history on our servers
                    </p>
                  </div>
                  <Switch 
                    id="storeHistory" 
                    checked={privacySettings.storeHistory}
                    onCheckedChange={(checked) => setPrivacySettings({...privacySettings, storeHistory: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="publicProfile">Public Profile</Label>
                    <p className="text-sm text-muted-foreground">
                      Allow others to see your profile and activity
                    </p>
                  </div>
                  <Switch 
                    id="publicProfile" 
                    checked={privacySettings.publicProfile}
                    onCheckedChange={(checked) => setPrivacySettings({...privacySettings, publicProfile: checked})}
                  />
                </div>
                
                <Separator />
                
                <div className="space-y-2">
                  <Label htmlFor="dataRetention">Data Retention Period</Label>
                  <Select 
                    value={privacySettings.dataRetention}
                    onValueChange={(value) => setPrivacySettings({...privacySettings, dataRetention: value})}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select retention period" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="30days">30 Days</SelectItem>
                      <SelectItem value="90days">90 Days</SelectItem>
                      <SelectItem value="1year">1 Year</SelectItem>
                      <SelectItem value="forever">Forever</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    How long we'll keep your data after your last activity
                  </p>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="cookiePreference">Cookie Preferences</Label>
                  <RadioGroup 
                    value={privacySettings.cookiePreference} 
                    onValueChange={(value) => setPrivacySettings({...privacySettings, cookiePreference: value})} 
                    className="flex flex-col space-y-1"
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="essential" id="cookies-essential" />
                      <Label htmlFor="cookies-essential">Essential Only</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="functional" id="cookies-functional" />
                      <Label htmlFor="cookies-functional">Functional</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="all" id="cookies-all" />
                      <Label htmlFor="cookies-all">All Cookies</Label>
                    </div>
                  </RadioGroup>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <Button variant="outline" className="w-full">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="7 10 12 15 17 10"/>
                      <line x1="12" x2="12" y1="15" y2="3"/>
                    </svg>
                    Download My Data
                  </Button>
                  
                  <Button variant="destructive" className="w-full">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                      <path d="M3 6h18"/>
                      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>
                      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>
                      <line x1="10" x2="10" y1="11" y2="17"/>
                      <line x1="14" x2="14" y1="11" y2="17"/>
                    </svg>
                    Delete All My Data
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Accessibility Tab */}
          <TabsContent value="accessibility">
            <Card>
              <CardHeader>
                <CardTitle>Accessibility Settings</CardTitle>
                <CardDescription>
                  Make the application more accessible based on your needs
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="highContrastMode">High Contrast Mode</Label>
                    <p className="text-sm text-muted-foreground">
                      Increase contrast for better visibility
                    </p>
                  </div>
                  <Switch 
                    id="highContrastMode" 
                    checked={appSettings.highContrastMode}
                    onCheckedChange={(checked) => setAppSettings({...appSettings, highContrastMode: checked})}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="reducedMotion">Reduced Motion</Label>
                    <p className="text-sm text-muted-foreground">
                      Minimize animations and motion effects
                    </p>
                  </div>
                  <Switch 
                    id="reducedMotion" 
                    checked={appSettings.reducedMotion}
                    onCheckedChange={(checked) => setAppSettings({...appSettings, reducedMotion: checked})}
                  />
                </div>
                
                <Separator />
                
                <div className="space-y-6">
                  <div className="space-y-2">
                    <Label>Text-to-Speech</Label>
                    <Button variant="outline" className="w-full">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                        <path d="M12 6v12"/>
                        <path d="M6 12h12"/>
                      </svg>
                      Configure Text-to-Speech
                    </Button>
                  </div>
                  
                  <div className="space-y-2">
                    <Label>Keyboard Shortcuts</Label>
                    <Button variant="outline" className="w-full">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                        <path d="M18 3a3 3 0 0 0-3 3v12a3 3 0 0 0 3 3 3 3 0 0 0 3-3 3 3 0 0 0-3-3H6a3 3 0 0 0-3 3 3 3 0 0 0 3 3 3 3 0 0 0 3-3V6a3 3 0 0 0-3-3 3 3 0 0 0-3 3 3 3 0 0 0 3 3h12a3 3 0 0 0 3-3 3 3 0 0 0-3-3z"/>
                      </svg>
                      Customize Keyboard Shortcuts
                    </Button>
                  </div>
                </div>
                
                <Separator />
                
                <div className="p-4 bg-muted rounded-md">
                  <h3 className="text-md font-medium mb-2">Accessibility Statement</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    We are committed to ensuring our platform is accessible to everyone. If you encounter any accessibility issues or have suggestions for improvement, please contact our support team.
                  </p>
                  <Button variant="outline" size="sm">
                    Contact Support
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Advanced Tab */}
          <TabsContent value="advanced">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>System Information</CardTitle>
                  <CardDescription>
                    Details about your system and the application
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium">Application Version</p>
                      <p className="text-sm text-muted-foreground">2.3.0</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Last Updated</p>
                      <p className="text-sm text-muted-foreground">May 15, 2025</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Browser</p>
                      <p className="text-sm text-muted-foreground">Chrome 118.0.5993.88</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Operating System</p>
                      <p className="text-sm text-muted-foreground">Windows 11</p>
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div className="space-y-2">
                    <Button variant="outline" className="w-full">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                      </svg>
                      Check for Updates
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Developer Options</CardTitle>
                  <CardDescription>
                    Advanced settings for developers
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="developerMode">Developer Mode</Label>
                      <p className="text-sm text-muted-foreground">
                        Enable advanced debugging features
                      </p>
                    </div>
                    <Switch id="developerMode" />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="verbose">Verbose Logging</Label>
                      <p className="text-sm text-muted-foreground">
                        Enable detailed application logs
                      </p>
                    </div>
                    <Switch id="verbose" />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="apiEndpoint">API Endpoint</Label>
                    <Input id="apiEndpoint" value="https://api.deepguard.ai/v1" />
                    <p className="text-sm text-muted-foreground">
                      Custom API endpoint for development purposes
                    </p>
                  </div>
                  
                  <Button variant="outline" className="w-full">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                      <path d="m21 21-6-6m6 6v-4.8m0 4.8h-4.8"/>
                      <path d="M3 16.2V21m0-4.8H7.8"/>
                      <path d="M16.2 3H21m-4.8 0V7.8"/>
                      <path d="M3 7.8V3m0 4.8H7.8"/>
                    </svg>
                    Open Developer Console
                  </Button>
                </CardContent>
              </Card>
              
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>Application Maintenance</CardTitle>
                  <CardDescription>
                    Maintain and troubleshoot the application
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Button variant="outline" className="w-full">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                        <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                        <path d="M3 3v5h5"/>
                        <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
                        <path d="M16 16h5v5"/>
                      </svg>
                      Reset Application
                    </Button>
                    
                    <Button variant="outline" className="w-full">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                        <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                        <polyline points="14 2 14 8 20 8"/>
                      </svg>
                      Export Logs
                    </Button>
                    
                    <Button variant="outline" className="w-full">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" x2="12" y1="8" y2="12"/>
                        <line x1="12" x2="12.01" y1="16" y2="16"/>
                      </svg>
                      Troubleshoot Issues
                    </Button>
                  </div>
                  
                  <Separator />
                  
                  <div className="p-4 bg-muted/50 rounded-md">
                    <h3 className="text-md font-medium mb-2">Cache & Storage</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Current storage usage: 24.3 MB
                    </p>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm">
                        Clear Cache
                      </Button>
                      <Button variant="outline" size="sm">
                        Manage Storage
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}