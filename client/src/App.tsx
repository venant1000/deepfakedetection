import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import HomePage from "@/pages/home-page";
import AuthPage from "@/pages/auth-page";
import DashboardPage from "@/pages/dashboard-page";
import AnalysisPage from "@/pages/analysis-page";
import UploadPage from "@/pages/upload-page";
import HistoryPage from "@/pages/history-page";
import ReportsPage from "@/pages/reports-page";
import ProfilePage from "@/pages/profile-page";
import SettingsPage from "@/pages/settings-page";
import HelpPage from "@/pages/help-page";
import AdminPage from "@/pages/admin-page";
import AdminUsersPage from "@/pages/admin-users-page";
import AdminAnalyticsPage from "@/pages/admin-analytics-page";
import AdminSettingsPage from "@/pages/admin-settings-page";
import NotFound from "@/pages/not-found";
import { ProtectedRoute } from "./lib/protected-route";
import ChatbotWidget from "./components/shared/chatbot-widget";

function Router() {
  return (
    <Switch>
      <Route path="/" component={HomePage} />
      <Route path="/auth" component={AuthPage} />
      <ProtectedRoute path="/dashboard" component={DashboardPage} />
      <ProtectedRoute path="/upload" component={UploadPage} />
      <ProtectedRoute path="/history" component={HistoryPage} />
      <ProtectedRoute path="/reports" component={ReportsPage} />
      <ProtectedRoute path="/profile" component={ProfilePage} />
      <ProtectedRoute path="/settings" component={SettingsPage} />
      <ProtectedRoute path="/help" component={HelpPage} />
      <ProtectedRoute path="/analysis/:id" component={AnalysisPage} />
      <ProtectedRoute path="/admin" component={AdminPage} />
      <ProtectedRoute path="/admin/users" component={AdminUsersPage} />
      <ProtectedRoute path="/admin/analytics" component={AdminAnalyticsPage} />
      <ProtectedRoute path="/admin/settings" component={AdminSettingsPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Router />
        <ChatbotWidget />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
