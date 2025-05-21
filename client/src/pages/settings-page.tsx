import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import Sidebar from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function SettingsPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  
  const [appSettings, setAppSettings] = useState({
    theme: "dark",
    autoplayVideos: false,
    highContrastMode: false,
    analysisConfidenceThreshold: 75,
    defaultVideoQuality: "auto",
    notifyOnAnalysisComplete: true,
    saveAnalysisHistory: true,
    dataPrivacyMode: "standard",
  });
  
  const handleSettingChange = (setting: string, value: any) => {
    setAppSettings(prev => ({
      ...prev,
      [setting]: value
    }));
    
    toast({
      title: "Settings updated",
      description: "Your preferences have been saved successfully.",
    });
  };

  const resetSettings = () => {
    setAppSettings({
      theme: "dark",
      autoplayVideos: false,
      highContrastMode: false,
      analysisConfidenceThreshold: 75,
      defaultVideoQuality: "auto",
      notifyOnAnalysisComplete: true,
      saveAnalysisHistory: true,
      dataPrivacyMode: "standard",
    });
    
    toast({
      title: "Settings reset",
      description: "All settings have been restored to their default values.",
    });
  };

  return (
    <div className="min-h-screen bg-background flex">
      <Sidebar isAdmin={user?.role === "admin"} />
      
      <div className="flex-1 ml-20 md:ml-64 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold">Application Settings</h1>
            <Button variant="outline" onClick={resetSettings}>Reset to Defaults</Button>
          </div>
          
          <div className="glass rounded-xl overflow-hidden">
            <div className="p-6 border-b border-muted">
              <h2 className="text-xl font-semibold">Appearance</h2>
            </div>
            
            <div className="p-6 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium mb-1">Application Theme</p>
                  <p className="text-sm text-muted-foreground">Choose your preferred visual style</p>
                </div>
                <Select 
                  value={appSettings.theme} 
                  onValueChange={(value) => handleSettingChange("theme", value)}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Select theme" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">Light</SelectItem>
                    <SelectItem value="dark">Dark</SelectItem>
                    <SelectItem value="system">System Default</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium mb-1">High Contrast Mode</p>
                  <p className="text-sm text-muted-foreground">Increase visual contrast for better readability</p>
                </div>
                <Switch 
                  checked={appSettings.highContrastMode} 
                  onCheckedChange={(value) => handleSettingChange("highContrastMode", value)}
                />
              </div>
            </div>
            
            <div className="p-6 border-t border-b border-muted">
              <h2 className="text-xl font-semibold">Video Settings</h2>
            </div>
            
            <div className="p-6 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium mb-1">Autoplay Videos</p>
                  <p className="text-sm text-muted-foreground">Automatically play videos when viewing analysis</p>
                </div>
                <Switch 
                  checked={appSettings.autoplayVideos} 
                  onCheckedChange={(value) => handleSettingChange("autoplayVideos", value)}
                />
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <p className="font-medium">Analysis Confidence Threshold</p>
                  <span className="text-sm text-muted-foreground">{appSettings.analysisConfidenceThreshold}%</span>
                </div>
                <Slider 
                  value={[appSettings.analysisConfidenceThreshold]} 
                  min={50}
                  max={95}
                  step={5}
                  onValueChange={(value) => handleSettingChange("analysisConfidenceThreshold", value[0])}
                  className="mb-1"
                />
                <p className="text-sm text-muted-foreground">Set the minimum confidence level for deepfake alerts</p>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium mb-1">Default Video Quality</p>
                  <p className="text-sm text-muted-foreground">Choose playback quality for analyzed videos</p>
                </div>
                <Select 
                  value={appSettings.defaultVideoQuality} 
                  onValueChange={(value) => handleSettingChange("defaultVideoQuality", value)}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Select quality" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low (480p)</SelectItem>
                    <SelectItem value="medium">Medium (720p)</SelectItem>
                    <SelectItem value="high">High (1080p)</SelectItem>
                    <SelectItem value="auto">Auto</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div className="p-6 border-t border-b border-muted">
              <h2 className="text-xl font-semibold">Privacy & Data</h2>
            </div>
            
            <div className="p-6 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium mb-1">Save Analysis History</p>
                  <p className="text-sm text-muted-foreground">Store results of previous video analyses</p>
                </div>
                <Switch 
                  checked={appSettings.saveAnalysisHistory} 
                  onCheckedChange={(value) => handleSettingChange("saveAnalysisHistory", value)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium mb-1">Analysis Complete Notifications</p>
                  <p className="text-sm text-muted-foreground">Receive alerts when video analysis is finished</p>
                </div>
                <Switch 
                  checked={appSettings.notifyOnAnalysisComplete} 
                  onCheckedChange={(value) => handleSettingChange("notifyOnAnalysisComplete", value)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium mb-1">Data Privacy Mode</p>
                  <p className="text-sm text-muted-foreground">Control how your data is processed and stored</p>
                </div>
                <Select 
                  value={appSettings.dataPrivacyMode} 
                  onValueChange={(value) => handleSettingChange("dataPrivacyMode", value)}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Select mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="standard">Standard</SelectItem>
                    <SelectItem value="enhanced">Enhanced Privacy</SelectItem>
                    <SelectItem value="maximum">Maximum Protection</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div className="p-6 border-t border-muted">
              <h2 className="text-xl font-semibold mb-4">Advanced Settings</h2>
              
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="apiSettings">
                  <AccordionTrigger>API Integration</AccordionTrigger>
                  <AccordionContent className="space-y-4">
                    <div>
                      <Label htmlFor="apiKey">Google Gemini API Key</Label>
                      <div className="flex gap-2 mt-2">
                        <input
                          type="password"
                          id="apiKey"
                          value="••••••••••••••••••••••••••••••"
                          disabled
                          className="flex-1 p-2 rounded-md border border-muted"
                        />
                        <Button variant="outline" size="sm">Update</Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Used for AI-powered deepfake detection
                      </p>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">Use Advanced AI Models</p>
                        <p className="text-xs text-muted-foreground">Enables higher accuracy but slower processing</p>
                      </div>
                      <Switch defaultChecked={true} />
                    </div>
                  </AccordionContent>
                </AccordionItem>
                
                <AccordionItem value="debugOptions">
                  <AccordionTrigger>Developer Options</AccordionTrigger>
                  <AccordionContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">Enable Debug Logging</p>
                        <p className="text-xs text-muted-foreground">Record detailed application logs</p>
                      </div>
                      <Switch defaultChecked={false} />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">Show Analysis Metrics</p>
                        <p className="text-xs text-muted-foreground">Display technical details in analysis results</p>
                      </div>
                      <Switch defaultChecked={true} />
                    </div>
                    
                    <Button variant="destructive" size="sm" className="mt-2">
                      Clear Application Cache
                    </Button>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}