import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { useAuth } from "../contexts/AuthContext";
import { User, Bell, Shield, CreditCard } from "lucide-react";

export function AppSettings() {
  const { user } = useAuth();

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 break-words">Settings</h1>
        <p className="text-slate-600 mt-2">
          Manage your account and preferences
        </p>
      </div>

      {/* Account Settings */}
      <Card className="p-6 mb-6">
        <div className="flex items-center gap-3 mb-6">
          <User className="h-5 w-5 text-slate-600" />
          <h2 className="text-xl font-semibold text-slate-900">Account</h2>
        </div>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Email
            </label>
            <Input value={user?.email || ""} disabled />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Display Name
            </label>
            <Input placeholder="Your name" />
          </div>
          <Button>Save Changes</Button>
        </div>
      </Card>

      {/* Notifications */}
      <Card className="p-6 mb-6">
        <div className="flex items-center gap-3 mb-6">
          <Bell className="h-5 w-5 text-slate-600" />
          <h2 className="text-xl font-semibold text-slate-900">Notifications</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-slate-900">Email notifications</p>
              <p className="text-sm text-slate-600">
                Receive updates about your documents
              </p>
            </div>
            <input type="checkbox" className="w-4 h-4" />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-slate-900">Processing complete</p>
              <p className="text-sm text-slate-600">
                Get notified when OCR is done
              </p>
            </div>
            <input type="checkbox" className="w-4 h-4" defaultChecked />
          </div>
        </div>
      </Card>

      {/* Security */}
      <Card className="p-6 mb-6">
        <div className="flex items-center gap-3 mb-6">
          <Shield className="h-5 w-5 text-slate-600" />
          <h2 className="text-xl font-semibold text-slate-900">Security</h2>
        </div>
        <div className="space-y-4">
          <Button variant="outline">Change Password</Button>
          <Button variant="outline" className="text-red-600 hover:text-red-700">
            Delete Account
          </Button>
        </div>
      </Card>

      {/* Billing */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-6">
          <CreditCard className="h-5 w-5 text-slate-600" />
          <h2 className="text-xl font-semibold text-slate-900">Billing</h2>
        </div>
        <div className="space-y-4">
          <div className="p-4 bg-slate-50 rounded-lg">
            <p className="font-medium text-slate-900">Free Plan</p>
            <p className="text-sm text-slate-600 mt-1">
              10 documents per month
            </p>
          </div>
          <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
            Upgrade to Pro
          </Button>
        </div>
      </Card>
    </div>
  );
}
