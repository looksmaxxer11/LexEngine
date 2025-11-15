import { Check, Sparkles } from "lucide-react";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";
import { motion } from "framer-motion";

const plans = [
  {
    name: "Starter",
    price: "49",
    description: "Perfect for small businesses and startups",
    features: [
      "1,000 documents/month",
      "99% accuracy guarantee",
      "Email support",
      "Basic API access",
      "5 users included",
      "Cloud storage (10GB)",
      "Standard processing speed"
    ],
    highlighted: false,
    cta: "Start Free Trial"
  },
  {
    name: "Professional",
    price: "149",
    description: "For growing teams with higher volume",
    features: [
      "10,000 documents/month",
      "99.9% accuracy guarantee",
      "Priority support",
      "Full API access",
      "20 users included",
      "Cloud storage (100GB)",
      "Fast processing speed",
      "Custom integrations",
      "Advanced analytics"
    ],
    highlighted: true,
    cta: "Start Free Trial",
    badge: "Most Popular"
  },
  {
    name: "Enterprise",
    price: "Custom",
    description: "For large organizations with custom needs",
    features: [
      "Unlimited documents",
      "99.99% accuracy guarantee",
      "24/7 dedicated support",
      "Full API + webhooks",
      "Unlimited users",
      "Custom storage solutions",
      "Fastest processing speed",
      "Custom integrations",
      "Advanced analytics + AI insights",
      "On-premise deployment option",
      "SLA guarantee",
      "Custom training"
    ],
    highlighted: false,
    cta: "Contact Sales"
  }
];

export function Pricing() {
  return (
    <section id="pricing" className="py-24 px-4 sm:px-6 lg:px-8 bg-slate-50 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute top-0 left-0 w-96 h-96 bg-blue-400/10 rounded-full blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
            scale: [1, 1.2, 1]
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute bottom-0 right-0 w-96 h-96 bg-purple-400/10 rounded-full blur-3xl"
          animate={{
            x: [0, -100, 0],
            y: [0, -50, 0],
            scale: [1, 1.3, 1]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
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
            Pricing
          </div>
          <h2 className="text-4xl sm:text-5xl mb-4">
            Simple, Transparent Pricing
          </h2>
          <p className="text-xl text-slate-600">
            Choose the perfect plan for your business. All plans include a 14-day free trial.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {plans.map((plan, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              whileHover={{ 
                y: -10,
                rotateY: plan.highlighted ? 0 : 5,
                scale: plan.highlighted ? 1 : 1.05
              }}
              style={{ transformStyle: "preserve-3d" }}
            >
              <Card 
                className={`relative p-8 bg-white transition-all duration-300 h-full flex flex-col ${
                  plan.highlighted 
                    ? 'border-2 border-blue-500 shadow-2xl scale-105 md:scale-110' 
                    : 'border-slate-200'
                }`}
              >
                {plan.badge && (
                  <Badge className="absolute -top-3 left-1/2 -translate-x-1/2 bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-1">
                    <Sparkles className="h-3 w-3 inline mr-1" />
                    {plan.badge}
                  </Badge>
                )}

                <div className="text-center mb-6">
                  <h3 className="text-2xl mb-2">{plan.name}</h3>
                  <p className="text-slate-600 text-sm mb-4">{plan.description}</p>
                  <div className="flex items-baseline justify-center gap-2">
                    {plan.price !== "Custom" && (
                      <span className="text-2xl text-slate-600">$</span>
                    )}
                    <span className="text-5xl">
                      {plan.price}
                    </span>
                    {plan.price !== "Custom" && (
                      <span className="text-slate-600">/month</span>
                    )}
                  </div>
                </div>

                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button 
                    className={`w-full mb-6 h-12 ${
                      plan.highlighted
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 shadow-lg shadow-blue-500/30'
                        : 'bg-slate-900 hover:bg-slate-800'
                    }`}
                  >
                    {plan.cta}
                  </Button>
                </motion.div>

                <div className="space-y-3 flex-1">
                  {plan.features.map((feature, featureIndex) => (
                    <motion.div 
                      key={featureIndex} 
                      className="flex items-start gap-3"
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: index * 0.1 + featureIndex * 0.05 }}
                    >
                      <div className={`mt-0.5 rounded-full p-1 ${
                        plan.highlighted 
                          ? 'bg-blue-100' 
                          : 'bg-slate-100'
                      }`}>
                        <Check className={`h-4 w-4 ${
                          plan.highlighted 
                            ? 'text-blue-600' 
                            : 'text-slate-600'
                        }`} />
                      </div>
                      <span className="text-slate-600 text-sm">{feature}</span>
                    </motion.div>
                  ))}
                </div>
              </Card>
            </motion.div>
          ))}
        </div>

        <motion.div 
          className="mt-12 text-center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
        >
          <p className="text-slate-600 mb-4">
            All plans include 14-day free trial • No credit card required • Cancel anytime
          </p>
          <p className="text-sm text-slate-500">
            Need a custom solution? <a href="#" className="text-blue-600 hover:underline">Contact our sales team</a>
          </p>
        </motion.div>
      </div>
    </section>
  );
}