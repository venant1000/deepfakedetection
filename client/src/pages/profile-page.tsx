import { useState, useEffect } from "react";
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
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { CheckCircle, AlertCircle, Upload, Loader2 } from "lucide-react";

export default function ProfilePage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  
  // Form state
  const [initialData, setInitialData] = useState({
    username: user?.username || "",
    fullName: user?.username || "User",
    email: "user@example.com",
    bio: "Deepfake detection enthusiast and tech advocate. Working to make the internet safer through AI-powered media analysis.",
    organization: "Media Truth Initiative",

    website: "https://example.com"
  });
  
  const [profileData, setProfileData] = useState({...initialData});

  // Load user data when available
  useEffect(() => {
    if (user) {
      const updatedData = {
        ...initialData,
        username: user.username,
        fullName: user.username // Use username as initial full name if not set yet
      };
      setInitialData(updatedData);
      setProfileData(updatedData);
    }
  }, [user]);
  
  // Security settings
  const [securitySettings, setSecuritySettings] = useState({
    twoFactorAuth: false,
    loginNotifications: true,
    sessionTimeout: 30 // minutes
  });

  // Track changes to detect if form has been modified
  useEffect(() => {
    if (JSON.stringify(profileData) !== JSON.stringify(initialData)) {
      setHasChanges(true);
      setIsSaved(false);
    } else {
      setHasChanges(false);
    }
  }, [profileData, initialData]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    
    // Clear error when field is edited
    if (formErrors[name]) {
      setFormErrors(prev => {
        const newErrors = {...prev};
        delete newErrors[name];
        return newErrors;
      });
    }
    
    setProfileData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const validateForm = (): boolean => {
    const errors: Record<string, string> = {};
    
    if (!profileData.fullName.trim()) {
      errors.fullName = "Full name is required";
    }
    
    if (!profileData.email.trim()) {
      errors.email = "Email is required";
    } else if (!/^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i.test(profileData.email)) {
      errors.email = "Invalid email address";
    }
    
    if (profileData.website && !/^(https?:\/\/)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$/i.test(profileData.website)) {
      errors.website = "Invalid website URL";
    }
    
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSaveProfile = async () => {
    if (!validateForm()) {
      toast({
        title: "Validation Error",
        description: "Please correct the errors in the form.",
        variant: "destructive"
      });
      return;
    }
    
    setIsLoading(true);
    
    try {
      // In a real app, this would be an API call to update the user profile
      // await apiRequest('/api/profile', {
      //   method: 'POST',
      //   body: JSON.stringify(profileData)
      // });
      
      // Simulate API call for now
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setInitialData(profileData);
      setIsSaved(true);
      setHasChanges(false);
      
      toast({
        title: "Profile updated",
        description: "Your profile information has been updated successfully."
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update profile. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadImage = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Check file size (max 2MB)
    if (file.size > 2 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Profile picture must be less than 2MB",
        variant: "destructive"
      });
      return;
    }
    
    // Check file type
    if (!['image/jpeg', 'image/png', 'image/gif'].includes(file.type)) {
      toast({
        title: "Invalid file type",
        description: "Please upload a JPG, PNG, or GIF image",
        variant: "destructive"
      });
      return;
    }
    
    setUploadLoading(true);
    
    // Simulate upload
    setTimeout(() => {
      setUploadLoading(false);
      toast({
        title: "Profile picture updated",
        description: "Your profile picture has been updated successfully."
      });
    }, 1500);
  };

  const handleChangePassword = () => {
    const currentPassword = (document.getElementById('currentPassword') as HTMLInputElement)?.value;
    const newPassword = (document.getElementById('newPassword') as HTMLInputElement)?.value;
    const confirmPassword = (document.getElementById('confirmPassword') as HTMLInputElement)?.value;
    
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast({
        title: "Missing fields",
        description: "Please fill in all password fields",
        variant: "destructive"
      });
      return;
    }
    
    if (newPassword.length < 8) {
      toast({
        title: "Password too short",
        description: "Password must be at least 8 characters long",
        variant: "destructive"
      });
      return;
    }
    
    if (newPassword !== confirmPassword) {
      toast({
        title: "Passwords don't match",
        description: "New password and confirmation don't match",
        variant: "destructive"
      });
      return;
    }
    
    // In a real app, this would be an API call to change the password
    
    // Clear password fields
    const inputs = document.querySelectorAll('input[type="password"]');
    inputs.forEach((input) => {
      (input as HTMLInputElement).value = '';
    });
    
    toast({
      title: "Password changed",
      description: "Your password has been updated successfully."
    });
  };

  const getInitials = (name: string) => {
    if (!name) return "U";
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
          {hasChanges && (
            <div className="flex items-center gap-2 text-amber-500 bg-amber-500/10 px-3 py-1 rounded-full text-xs">
              <AlertCircle size={14} />
              <span>You have unsaved changes</span>
            </div>
          )}
          {isSaved && (
            <div className="flex items-center gap-2 text-green-500 bg-green-500/10 px-3 py-1 rounded-full text-xs">
              <CheckCircle size={14} />
              <span>Profile saved successfully</span>
            </div>
          )}
        </div>

        <Tabs defaultValue="profile" className="w-full">
          <TabsList className="mb-6 w-full md:w-auto">
            <TabsTrigger value="profile">Profile Information</TabsTrigger>
            <TabsTrigger value="security">Security</TabsTrigger>
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
                        <Label htmlFor="fullName" className="flex items-center">
                          Full Name
                          {formErrors.fullName && <span className="text-red-500 ml-2 text-xs">*Required</span>}
                        </Label>
                        <Input 
                          id="fullName" 
                          name="fullName"
                          value={profileData.fullName}
                          onChange={handleInputChange}
                          className={formErrors.fullName ? "border-red-500 focus:ring-red-500" : ""}
                        />
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <Label htmlFor="email" className="flex items-center">
                        Email Address
                        {formErrors.email && <span className="text-red-500 ml-2 text-xs">{formErrors.email}</span>}
                      </Label>
                      <Input 
                        id="email" 
                        name="email"
                        type="email"
                        value={profileData.email}
                        onChange={handleInputChange}
                        className={formErrors.email ? "border-red-500 focus:ring-red-500" : ""}
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
                      <Label htmlFor="website" className="flex items-center">
                        Website
                        {formErrors.website && <span className="text-red-500 ml-2 text-xs">{formErrors.website}</span>}
                      </Label>
                      <Input 
                        id="website" 
                        name="website"
                        value={profileData.website}
                        onChange={handleInputChange}
                        className={formErrors.website ? "border-red-500 focus:ring-red-500" : ""}
                      />
                    </div>
                  </CardContent>
                  <CardFooter className="flex justify-between border-t pt-6">
                    <Button 
                      variant="outline" 
                      onClick={() => {
                        setProfileData(initialData);
                        setFormErrors({});
                        setHasChanges(false);
                      }}
                      disabled={!hasChanges || isLoading}
                    >
                      Reset Changes
                    </Button>
                    <Button 
                      onClick={handleSaveProfile} 
                      disabled={isLoading || !hasChanges}
                      className={!hasChanges ? "opacity-70" : ""}
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Saving...
                        </>
                      ) : isSaved ? (
                        <>
                          <CheckCircle className="mr-2 h-4 w-4" />
                          Saved
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
                      <label className="relative">
                        <input 
                          type="file"
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                          accept="image/jpeg, image/png, image/gif"
                          onChange={handleUploadImage}
                          disabled={uploadLoading}
                        />
                        <Button variant="outline" className="pointer-events-none">
                          {uploadLoading ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              Uploading...
                            </>
                          ) : (
                            <>
                              <Upload className="mr-2 h-4 w-4" />
                              Upload New
                            </>
                          )}
                        </Button>
                      </label>
                      <Button variant="destructive" disabled={uploadLoading}>
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
                        <div className="h-3 w-3 rounded-full bg-green-500"></div>
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
                    <CardTitle>Login Activity</CardTitle>
                    <CardDescription>
                      Recent login sessions
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="font-medium">Current Session</p>
                          <p className="text-sm text-muted-foreground">San Francisco, CA • Chrome on Windows</p>
                        </div>
                        <Badge variant="outline" className="bg-green-500 text-white">Active</Badge>
                      </div>
                      <Separator />
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="font-medium">Yesterday, 3:42 PM</p>
                          <p className="text-sm text-muted-foreground">San Francisco, CA • Chrome on Windows</p>
                        </div>
                        <Badge variant="outline">Expired</Badge>
                      </div>
                      <Separator />
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="font-medium">May 19, 10:15 AM</p>
                          <p className="text-sm text-muted-foreground">San Francisco, CA • Firefox on MacOS</p>
                        </div>
                        <Badge variant="outline">Expired</Badge>
                      </div>
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
                          Receive alerts for new login attempts
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
        </Tabs>
      </div>
    </div>
  );
}