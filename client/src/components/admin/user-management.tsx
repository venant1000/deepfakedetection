import { useState } from "react";
import { Button } from "@/components/ui/button";

// Mock user data for demonstration purposes
const mockUsers = [
  {
    id: 1,
    name: "Alex Morgan",
    email: "alex.morgan@example.com",
    role: "verified",
    status: "active",
    lastActivity: "10 minutes ago"
  },
  {
    id: 2,
    name: "Sarah Chen",
    email: "sarah.chen@example.com",
    role: "admin",
    status: "active",
    lastActivity: "2 hours ago"
  },
  {
    id: 3,
    name: "James Wilson",
    email: "james.wilson@example.com",
    role: "user",
    status: "suspended",
    lastActivity: "5 days ago"
  },
  {
    id: 4,
    name: "Olivia Johnson",
    email: "olivia.johnson@example.com",
    role: "moderator",
    status: "active",
    lastActivity: "1 day ago"
  }
];

export default function UserManagement() {
  const [searchTerm, setSearchTerm] = useState("");
  const [roleFilter, setRoleFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [currentPage, setCurrentPage] = useState(1);
  const usersPerPage = 4;
  
  // Filter users based on search term and filters
  const filteredUsers = mockUsers.filter(user => {
    const matchesSearch = user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = roleFilter === "all" || user.role === roleFilter;
    const matchesStatus = statusFilter === "all" || user.status === statusFilter;
    
    return matchesSearch && matchesRole && matchesStatus;
  });
  
  // Pagination
  const totalPages = Math.ceil(filteredUsers.length / usersPerPage);
  const indexOfLastUser = currentPage * usersPerPage;
  const indexOfFirstUser = indexOfLastUser - usersPerPage;
  const currentUsers = filteredUsers.slice(indexOfFirstUser, indexOfLastUser);
  
  // Role badge styling
  const getRoleBadgeClass = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-purple-500/20 text-purple-400';
      case 'moderator':
        return 'bg-blue-500/20 text-blue-400';
      case 'verified':
        return 'bg-primary/20 text-primary';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };
  
  // Status badge styling
  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-primary/20 text-primary';
      case 'suspended':
        return 'bg-[#ffbb00]/20 text-[#ffbb00]';
      case 'pending':
        return 'bg-blue-500/20 text-blue-400';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };
  
  return (
    <div className="glass rounded-xl p-6 mb-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold">User Management</h2>
        <Button className="py-2 px-4 rounded-lg bg-primary text-black font-medium hover:opacity-90 transition-all flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14" />
            <path d="M5 12h14" />
          </svg>
          <span>Add User</span>
        </Button>
      </div>
      
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="relative">
          <input 
            type="text" 
            placeholder="Search users..." 
            className="py-2 px-4 pl-10 rounded-lg glass-dark border border-muted focus:outline-none focus:border-primary transition-colors md:w-64 w-full"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)} 
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
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
        </div>
        
        <select 
          className="py-2 px-4 rounded-lg glass-dark border border-muted focus:outline-none focus:border-primary transition-colors"
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
        >
          <option value="all">All Roles</option>
          <option value="admin">Admin</option>
          <option value="moderator">Moderator</option>
          <option value="premium">Premium</option>
          <option value="free">Free</option>
        </select>
        
        <select 
          className="py-2 px-4 rounded-lg glass-dark border border-muted focus:outline-none focus:border-primary transition-colors"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="all">All Status</option>
          <option value="active">Active</option>
          <option value="suspended">Suspended</option>
          <option value="pending">Pending</option>
        </select>
      </div>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-muted">
              <th className="pb-3 text-left text-muted-foreground font-medium">User</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Email</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Role</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Status</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Last Activity</th>
              <th className="pb-3 text-left text-muted-foreground font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {currentUsers.map((user) => (
              <tr key={user.id} className="border-b border-muted">
                <td className="py-4">
                  <div className="flex items-center gap-3">
                    <div className="h-9 w-9 rounded-full bg-muted flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
                        <circle cx="12" cy="7" r="4" />
                      </svg>
                    </div>
                    <span>{user.name}</span>
                  </div>
                </td>
                <td className="py-4 text-muted-foreground">{user.email}</td>
                <td className="py-4">
                  <span className={`py-1 px-3 rounded-full text-sm ${getRoleBadgeClass(user.role)}`}>
                    {user.role === 'premium' ? 'Premium User' : 
                     user.role === 'free' ? 'Free User' : 
                     user.role.charAt(0).toUpperCase() + user.role.slice(1)}
                  </span>
                </td>
                <td className="py-4">
                  <span className={`py-1 px-3 rounded-full text-sm ${getStatusBadgeClass(user.status)}`}>
                    {user.status.charAt(0).toUpperCase() + user.status.slice(1)}
                  </span>
                </td>
                <td className="py-4 text-muted-foreground">{user.lastActivity}</td>
                <td className="py-4">
                  <div className="flex items-center gap-2">
                    <Button variant="ghost" size="icon" className="p-2 rounded-lg glass-dark text-muted-foreground hover:text-white transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
                        <path d="m15 5 4 4" />
                      </svg>
                    </Button>
                    <Button variant="ghost" size="icon" className="p-2 rounded-lg glass-dark text-muted-foreground hover:text-white transition-colors">
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 6h18" />
                        <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                        <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                        <line x1="10" x2="10" y1="11" y2="17" />
                        <line x1="14" x2="14" y1="11" y2="17" />
                      </svg>
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="flex justify-between items-center mt-6">
        <div className="text-sm text-muted-foreground">
          Showing {indexOfFirstUser + 1}-{Math.min(indexOfLastUser, filteredUsers.length)} of {filteredUsers.length} users
        </div>
        <div className="flex items-center gap-2">
          <Button 
            variant="ghost" 
            size="icon" 
            className="p-2 rounded-lg glass-dark text-muted-foreground hover:text-white transition-colors"
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m15 18-6-6 6-6" />
            </svg>
          </Button>
          
          {[...Array(Math.min(totalPages, 3))].map((_, i) => (
            <Button 
              key={i}
              variant={currentPage === i + 1 ? "default" : "ghost"}
              className={`p-2 rounded-lg ${currentPage === i + 1 ? 'bg-muted text-white' : 'glass-dark text-muted-foreground hover:text-white transition-colors'}`}
              onClick={() => setCurrentPage(i + 1)}
            >
              {i + 1}
            </Button>
          ))}
          
          <Button 
            variant="ghost" 
            size="icon" 
            className="p-2 rounded-lg glass-dark text-muted-foreground hover:text-white transition-colors"
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m9 18 6-6-6-6" />
            </svg>
          </Button>
        </div>
      </div>
    </div>
  );
}
