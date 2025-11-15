import { motion } from "motion/react";
import { Card } from "./ui/card";
import { Brain, Cpu, Zap, Shield, Code, Sparkles } from "lucide-react";
import { Badge } from "./ui/badge";

const aiModels = [
  {
    name: "Tesseract OCR",
    description: "Open-source OCR engine with exceptional accuracy for printed text",
    icon: Code,
    features: ["100+ Languages", "Pattern Recognition", "Layout Analysis"],
    color: "from-blue-500 to-cyan-500"
  },
  {
    name: "Qwen-VL",
    description: "Advanced AI vision-language model for complex document understanding",
    icon: Brain,
    features: ["Deep Learning", "Context Understanding", "Multi-modal Processing"],
    color: "from-purple-500 to-pink-500"
  }
];

const capabilities = [
  { icon: Zap, label: "Real-time Processing", value: "<2s" },
  { icon: Shield, label: "99.9% Accuracy", value: "Guaranteed" },
  { icon: Cpu, label: "AI-Powered", value: "Deep Learning" },
  { icon: Sparkles, label: "Auto-Enhancement", value: "Smart Correction" }
];

export function AIShowcase() {
  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 bg-slate-900 relative overflow-hidden">
      {/* Animated background grid */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0" style={{
          backgroundImage: `linear-gradient(rgba(59, 130, 246, 0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(59, 130, 246, 0.3) 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }}></div>
      </div>

      {/* Floating particles */}
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-2 h-2 bg-blue-400 rounded-full"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [0.2, 0.8, 0.2],
            scale: [1, 1.5, 1]
          }}
          transition={{
            duration: 3 + Math.random() * 2,
            repeat: Infinity,
            delay: Math.random() * 2
          }}
        />
      ))}

      <div className="container mx-auto relative z-10">
        <motion.div 
          className="text-center max-w-3xl mx-auto mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <Badge className="inline-flex items-center gap-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 rounded-full mb-4 border-0">
            <Brain className="h-4 w-4" />
            <span>AI Technology</span>
          </Badge>
          <h2 className="text-4xl sm:text-5xl mb-4 text-white">
            Powered by Advanced AI Models
          </h2>
          <p className="text-xl text-slate-300">
            Combining the best of open-source and cutting-edge AI for unmatched accuracy
          </p>
        </motion.div>

        {/* AI Models Grid */}
        <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto mb-12">
          {aiModels.map((model, index) => {
            const Icon = model.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                whileHover={{ 
                  scale: 1.05,
                  rotateY: 5,
                  rotateX: 5,
                }}
                style={{ transformStyle: "preserve-3d" }}
              >
                <Card className="p-8 bg-slate-800/50 backdrop-blur-sm border-slate-700 hover:border-blue-500 transition-all duration-300 h-full">
                  <div className="flex items-start gap-4 mb-6">
                    <motion.div 
                      className={`bg-gradient-to-br ${model.color} w-16 h-16 rounded-2xl flex items-center justify-center shadow-2xl`}
                      animate={{ 
                        rotateY: [0, 360],
                      }}
                      transition={{ 
                        duration: 10,
                        repeat: Infinity,
                        ease: "linear"
                      }}
                    >
                      <Icon className="h-8 w-8 text-white" />
                    </motion.div>
                    <div className="flex-1">
                      <h3 className="text-2xl mb-2 text-white">{model.name}</h3>
                      <p className="text-slate-400">{model.description}</p>
                    </div>
                  </div>
                  
                  <div className="flex flex-wrap gap-2">
                    {model.features.map((feature, i) => (
                      <Badge 
                        key={i}
                        className="bg-slate-700/50 text-slate-300 border-slate-600 hover:bg-slate-600/50"
                      >
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Capabilities */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-5xl mx-auto">
          {capabilities.map((capability, index) => {
            const Icon = capability.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                whileHover={{ 
                  scale: 1.1,
                  rotateZ: 5
                }}
              >
                <Card className="p-6 bg-slate-800/30 backdrop-blur-sm border-slate-700/50 text-center hover:bg-slate-800/50 transition-all">
                  <div className="bg-gradient-to-br from-blue-500 to-purple-500 w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-3 shadow-lg">
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <div className="text-2xl text-white mb-1">{capability.value}</div>
                  <div className="text-sm text-slate-400">{capability.label}</div>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {/* Code snippet showcase */}
        <motion.div 
          className="mt-12 max-w-3xl mx-auto"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <Card className="p-6 bg-slate-800/50 backdrop-blur-sm border-slate-700">
            <div className="flex items-center gap-2 mb-4">
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <span className="text-slate-400 text-sm ml-2">Example API Request</span>
            </div>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm">
                <code>
                  <span className="text-purple-400">curl</span> <span className="text-blue-400">-X POST</span> <span className="text-green-400">"https://api.ocrify.com/v1/process"</span> \
                  {'\n'}  <span className="text-blue-400">-H</span> <span className="text-green-400">"Authorization: Bearer YOUR_API_KEY"</span> \
                  {'\n'}  <span className="text-blue-400">-F</span> <span className="text-green-400">"file=@document.pdf"</span> \
                  {'\n'}  <span className="text-blue-400">-F</span> <span className="text-green-400">"model=qwen-vl"</span>
                </code>
              </pre>
            </div>
          </Card>
        </motion.div>
      </div>
    </section>
  );
}
