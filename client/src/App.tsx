import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import HomePage from "@/pages/home-page";
import AuthPage from "@/pages/auth-page";
import DashboardPage from "@/pages/dashboard-page";
import AnalysisPage from "@/pages/analysis-page";
import UploadPage from "@/pages/upload-page.jsx";
import AdminPage from "@/pages/admin-page";
import AdminUsersPage from "@/pages/admin-users-page.jsx";
import AdminAnalyticsPage from "@/pages/admin-analytics-page.jsx";
import AdminLogsPage from "@/pages/admin-logs-page";
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
      <ProtectedRoute path="/analysis/:id" component={AnalysisPage} />
      <ProtectedRoute path="/admin" component={AdminPage} />
      <ProtectedRoute path="/admin/users" component={AdminUsersPage} />
      <ProtectedRoute path="/admin/analytics" component={AdminAnalyticsPage} />
      <ProtectedRoute path="/admin/logs" component={AdminLogsPage} />
      <ProtectedRoute path="/admin/settings" component={AdminSettingsPage} />
      <ProtectedRoute path="/reports" component={ReportsPage} />
      <ProtectedRoute path="/profile" component={ProfilePage} />
      <ProtectedRoute path="/settings" component={SettingsPage} />
      <ProtectedRoute path="/help" component={HelpPage} />
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
