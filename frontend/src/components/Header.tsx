import { Button } from "./ui/button";
import { ScanText, Menu } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { useState } from "react";
import { AuthModal } from "./AuthModal";
import { Link } from "react-router-dom";

export function Header() {
  const { user } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-lg shadow-lg">
                <ScanText className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                OCRify
              </span>
            </div>
            
            <nav className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-slate-600 hover:text-slate-900 transition-colors">
                Features
              </a>
              <a href="#how-it-works" className="text-slate-600 hover:text-slate-900 transition-colors">
                How It Works
              </a>
              <a href="#pricing" className="text-slate-600 hover:text-slate-900 transition-colors">
                Pricing
              </a>
            </nav>

            <div className="flex items-center gap-3">
              {user ? (
                <Link to="/app">
                  <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-lg shadow-blue-500/30">
                    Go to Dashboard
                  </Button>
                </Link>
              ) : (
                <>
                  <Button
                    onClick={() => setShowAuthModal(true)}
                    variant="ghost"
                    className="hidden sm:inline-flex"
                  >
                    Sign In
                  </Button>
                  <Button
                    onClick={() => setShowAuthModal(true)}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-lg shadow-blue-500/30"
                  >
                    Get Started
                  </Button>
                </>
              )}
              <Button variant="ghost" size="icon" className="md:hidden">
                <Menu className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {showAuthModal && (
        <AuthModal onClose={() => setShowAuthModal(false)} redirectTo="/app" />
      )}
    </>
  );
}