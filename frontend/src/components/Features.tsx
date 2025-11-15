import { 
  Zap, 
  Globe, 
  Lock, 
  FileText, 
  BarChart3, 
  Cloud,
  CheckCircle,
  RefreshCw
} from "lucide-react";
import { Card } from "./ui/card";
import { motion } from "framer-motion";

const features = [
  {
    icon: Zap,
    title: "Lightning Fast",
    description: "Process thousands of documents in minutes with our optimized OCR engine.",
    color: "from-yellow-500 to-orange-500"
  },
  {
    icon: Globe,
    title: "Multi-Language Support",
    description: "Recognize text in over 100 languages with exceptional accuracy.",
    color: "from-blue-500 to-cyan-500"
  },
  {
    icon: Lock,
    title: "Enterprise Security",
    description: "Bank-level encryption and compliance with SOC 2, GDPR, and HIPAA standards.",
    color: "from-green-500 to-emerald-500"
  },
  {
    icon: FileText,
    title: "Multiple Formats",
    description: "Support for PDF, images, scanned documents, and handwritten text.",
    color: "from-purple-500 to-pink-500"
  },
  {
    icon: BarChart3,
    title: "Advanced Analytics",
    description: "Track processing metrics and gain insights from your document data.",
    color: "from-indigo-500 to-purple-500"
  },
  {
    icon: Cloud,
    title: "Cloud & On-Premise",
    description: "Deploy in the cloud or on your own infrastructure for maximum control.",
    color: "from-sky-500 to-blue-500"
  },
  {
    icon: CheckCircle,
    title: "High Accuracy",
    description: "Industry-leading 99.9% accuracy with machine learning improvements.",
    color: "from-teal-500 to-green-500"
  },
  {
    icon: RefreshCw,
    title: "API Integration",
    description: "Seamlessly integrate with your existing workflows via REST API.",
    color: "from-rose-500 to-red-500"
  }
];

export function Features() {
  return (
    <section id="features" className="py-24 px-4 sm:px-6 lg:px-8 bg-slate-50 relative overflow-hidden">
      {/* Animated background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgb(99, 102, 241) 1px, transparent 0)`,
          backgroundSize: '40px 40px'
        }}></div>
      </div>

      <div className="container mx-auto relative z-10">
        <motion.div 
          className="text-center max-w-3xl mx-auto mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-block bg-blue-100 text-blue-700 px-4 py-1 rounded-full mb-4">
            Features
          </div>
          <h2 className="text-4xl sm:text-5xl mb-4">
            Everything You Need for Document Automation
          </h2>
          <p className="text-xl text-slate-600">
            Powerful features designed to streamline your document processing workflow
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{ 
                  y: -10,
                  rotateY: 5,
                  rotateX: 5
                }}
                style={{ transformStyle: "preserve-3d" }}
              >
                <Card 
                  className="p-6 bg-white hover:shadow-xl transition-all duration-300 border-slate-200 group h-full"
                >
                  <motion.div 
                    className={`bg-gradient-to-br ${feature.color} w-12 h-12 rounded-xl flex items-center justify-center mb-4 shadow-lg`}
                    animate={{
                      rotateY: [0, 360]
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "linear",
                      delay: index * 0.2
                    }}
                  >
                    <Icon className="h-6 w-6 text-white" />
                  </motion.div>
                  <h3 className="mb-2">{feature.title}</h3>
                  <p className="text-slate-600">{feature.description}</p>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}