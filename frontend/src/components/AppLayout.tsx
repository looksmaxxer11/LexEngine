import { useState } from "react";
import { Link, useLocation, Outlet } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import {
  LayoutDashboard,
  Upload,
  FileText,
  Settings,
  LogOut,
  Menu,
  X,
  ScanText,
} from "lucide-react";
import { Button } from "./ui/button";

export function AppLayout() {
  const { user, signOut } = useAuth();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const navigation = [
    { name: "Overview", href: "/app", icon: LayoutDashboard },
    { name: "Upload", href: "/app/upload", icon: Upload },
    { name: "History", href: "/app/history", icon: FileText },
    { name: "Settings", href: "/app/settings", icon: Settings },
  ];

  const isActive = (path: string) => {
    if (path === "/app") {
      return location.pathname === "/app";
    }
    return location.pathname.startsWith(path);
  };

  return (
    <div className="h-screen flex bg-slate-50 overflow-hidden">
      {/* Sidebar (now always static, non-floating) */}
      <aside
        className="w-64 bg-white border-r border-slate-200 shadow-sm h-full"
      >
        <div className="h-full flex flex-col">
          {/* Logo */}
          <div className="h-16 flex items-center justify-between px-6 border-b border-slate-200">
            <Link to="/" className="flex items-center gap-2">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-lg">
                <ScanText className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                OCRify
              </span>
            </Link>
            {/* Close button removed for static sidebar */}
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => {
              const active = isActive(item.href);
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`
                    flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium
                    transition-colors
                    ${
                      active
                        ? "bg-gradient-to-r from-blue-50 to-purple-50 text-blue-600"
                        : "text-slate-600 hover:bg-slate-50"
                    }
                  `}
                >
                  <item.icon className="h-5 w-5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* User section */}
          <div className="p-4 border-t border-slate-200">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 text-white flex items-center justify-center text-sm font-semibold">
                {user?.email?.[0]?.toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-slate-900 truncate">
                  {user?.email?.split("@")[0]}
                </p>
                <p className="text-xs text-slate-500 truncate">{user?.email}</p>
              </div>
            </div>
            <Button
              onClick={signOut}
              variant="outline"
              size="sm"
              className="w-full justify-start gap-2 text-slate-600 hover:text-red-600 hover:border-red-200"
            >
              <LogOut className="h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col h-full">
        {/* Top bar */}
        <header className="h-16 box-border bg-white border-b border-slate-200 flex items-center justify-between px-4 lg:px-6 shadow-sm">
          {/* Mobile menu button removed since sidebar is static */}
          <div className="flex-1" />
          <div className="flex items-center gap-3">
            <Link
              to="/"
              className="text-sm text-slate-600 hover:text-slate-900"
            >
              Back to Home
            </Link>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 bg-slate-50 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
