import { Header } from "./components/Header";
import { Footer } from "./components/Footer";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { Routes, Route, Navigate } from "react-router-dom";
import { Hero } from "./components/Hero";
import { Stats } from "./components/Stats";
import { Features } from "./components/Features";
import { AIShowcase } from "./components/AIShowcase";
import { HowItWorks } from "./components/HowItWorks";
import { UseCases } from "./components/UseCases";
import { Integrations } from "./components/Integrations";
import { Testimonials } from "./components/Testimonials";
import { Pricing } from "./components/Pricing";
import { CTA } from "./components/CTA";
import { AuthModal } from "./components/AuthModal";
import { AppLayout } from "./components/AppLayout";
import { AppOverview } from "./pages/AppOverview";
import { AppUpload } from "./pages/AppUpload";
import { AppHistory } from "./pages/AppHistory";
import { AppSettings } from "./pages/AppSettings";

function PublicLanding() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <Header />
      <main className="pt-16">
        <Hero />
        <Stats />
        {/* Upload now gated behind auth; use CTA in header/profile menu */}
        <Features />
        <AIShowcase />
        <HowItWorks />
        <UseCases />
        <Integrations />
        <Testimonials />
        <Pricing />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}

function AuthPage() {
  // Render the modal centered as a page
  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center">
      <AuthModal onClose={() => (window.location.href = "/")} />
    </div>
  );
}

function ProtectedRoute({ children }: { children: JSX.Element }) {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-slate-600">Loading...</p>
        </div>
      </div>
    );
  }
  if (!user) return <Navigate to="/auth" replace />;
  return children;
}

function AppRoutes() {
  return (
    <Routes>
      {/* Public marketing site */}
      <Route path="/" element={<PublicLanding />} />
      
      {/* Auth page */}
      <Route path="/auth" element={<AuthPage />} />
      
      {/* Protected dashboard app with layout */}
      <Route
        path="/app"
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<AppOverview />} />
        <Route path="upload" element={<AppUpload />} />
        <Route path="history" element={<AppHistory />} />
        <Route path="settings" element={<AppSettings />} />
      </Route>
      
      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}