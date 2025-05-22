// SVG Illustration generation functions

function renderHeroIllustration() {
    const container = document.getElementById('heroIllustration');
    if (!container) return;
    // Agent Perception-Decision-Action cycle - Outline style for Hero
    container.innerHTML = `
    <svg width="100%" height="100%" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
        <style>
            .hero-node { fill: rgba(240, 94, 28, 0.03); stroke: #F05E1C; stroke-width: 1.5; transition: all 0.3s ease; }
            .hero-node-core { fill: rgba(240, 94, 28, 0.08); stroke: #F05E1C; stroke-width: 2; }
            .hero-text { font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 500; fill: #4A5568; text-anchor: middle; dominant-baseline: central; }
            .hero-text-core { font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600; fill: #F05E1C; text-anchor: middle; dominant-baseline: central; }
            .hero-arrow { stroke: #F05E1C; stroke-width: 1.5; marker-end: url(#hero-arrowhead); fill: none; opacity: 0.8; }
            .hero-arrow-outer { stroke: rgba(240, 94, 28, 0.3); stroke-width: 1.5; marker-end: url(#hero-arrowhead-outer); fill: none; stroke-dasharray: 3 3; }
            #hero-arrowhead { fill: #F05E1C; }
            #hero-arrowhead-outer { fill: rgba(240, 94, 28, 0.5); }
        </style>
        <defs>
            <marker id="hero-arrowhead" orient="auto-start-reverse" markerWidth="8" markerHeight="8" refX="7" refY="4">
                <path d="M0,0 L8,4 L0,8 Z" />
            </marker>
            <marker id="hero-arrowhead-outer" orient="auto-start-reverse" markerWidth="6" markerHeight="6" refX="5" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" />
            </marker>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>

        <!-- Central Agent Brain -->
        <circle cx="200" cy="150" r="60" class="hero-node-core" filter="url(#glow)"/>
        <text x="200" y="145" class="hero-text-core">Agent Core</text>
        <text x="200" y="162" class="hero-text" style="font-size:10px; fill: #D0501A;">Decision Engine</text>

        <!-- Perception -->
        <g transform="translate(50, 70)">
            <rect x="0" y="0" width="80" height="50" rx="8" class="hero-node"/>
            <text x="40" y="25" class="hero-text">Perception</text>
        </g>
        <path d="M130,95 Q150,110 165,128" class="hero-arrow"/>

        <!-- Action -->
        <g transform="translate(270, 200)">
             <rect x="0" y="0" width="80" height="50" rx="8" class="hero-node"/>
            <text x="40" y="25" class="hero-text">Action</text>
        </g>
        <path d="M235,172 Q250,190 270,205" class="hero-arrow"/>
        
        <!-- Environment Interaction Loop (Outer) -->
        <path d="M270,225 C350,220 380,150 350,80 C320,10 80,10 50,95 C20,180 130,250 200,240" class="hero-arrow-outer" />
        <text x="200" y="35" class="hero-text" style="font-size:10px;">Environment</text>
        
        <!-- Data/Signal icons -->
        <circle cx="70" cy="150" r="8" class="hero-node" style="fill: rgba(240, 94, 28, 0.1);" />
        <circle cx="330" cy="130" r="8" class="hero-node" style="fill: rgba(240, 94, 28, 0.1);" />
        <circle cx="200" cy="230" r="6" class="hero-node" style="fill: rgba(240, 94, 28, 0.05);" />
        <circle cx="150" cy="50" r="6" class="hero-node" style="fill: rgba(240, 94, 28, 0.05);" />
    </svg>
    `;
}

function createAgentCycleIllustration() {
    return `
    <svg width="100%" viewBox="0 0 500 180" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
        <style>
            .cycle-node { fill: rgba(240, 94, 28, 0.05); stroke: #F05E1C; stroke-width: 1.5; }
            .cycle-text { font-family: 'Inter', sans-serif; font-size: 13px; text-anchor: middle; dominant-baseline: central; fill: #333; font-weight: 500; }
            .cycle-arrow { stroke: #F05E1C; stroke-width: 1.5; marker-end: url(#cycle-arrowhead); fill: none; }
            .env-text { font-family: 'Inter', sans-serif; font-size: 11px; fill: #6B7280; text-anchor: middle; }
            #cycle-arrowhead { fill: #F05E1C; }
        </style>
        <defs>
            <marker id="cycle-arrowhead" orient="auto-start-reverse" markerWidth="7" markerHeight="7" refX="6" refY="3.5">
                <path d="M0,0 L7,3.5 L0,7 Z" />
            </marker>
        </defs>

        <rect class="cycle-node" x="30" y="60" width="120" height="50" rx="8"/>
        <text class="cycle-text" x="90" y="85">Perception</text>

        <rect class="cycle-node" x="190" y="60" width="120" height="50" rx="8"/>
        <text class="cycle-text" x="250" y="85">Decision</text>

        <rect class="cycle-node" x="350" y="60" width="120" height="50" rx="8"/>
        <text class="cycle-text" x="410" y="85">Action</text>

        <path class="cycle-arrow" d="M150,85 L190,85"/>
        <path class="cycle-arrow" d="M310,85 L350,85"/>
        
        <path class="cycle-arrow" d="M470,85 C490,85 490,35 410,35 L90,35 C10,35 10,85 30,85" />
        <text class="env-text" x="250" y="15">Environment Interaction & Feedback Loop</text>
    </svg>
    `;
}

function createTransformerIllustration() {
    return `
    <svg width="100%" viewBox="0 0 600 350" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
        <style>
            .tf-block { fill: rgba(240, 94, 28, 0.03); stroke: #F05E1C; stroke-width: 1px; rx: 8px; transition: all 0.2s ease-in-out; }
            .tf-block:hover { fill: rgba(240, 94, 28, 0.08); }
            .tf-sub-block { fill: rgba(240, 94, 28, 0.08); stroke: #F05E1C; stroke-width: 1px; rx: 4px; }
            .tf-text { font-family: 'Inter', sans-serif; font-size: 11px; fill: #333; text-anchor: middle; dominant-baseline: central; font-weight: 500;}
            .tf-text-sm { font-size: 9px; fill: #555; }
            .tf-arrow { stroke: #F05E1C; stroke-width: 1px; marker-end: url(#tf-arrowhead); fill: none; opacity: 0.9; }
            #tf-arrowhead { fill: #F05E1C; }
        </style>
        <defs>
            <marker id="tf-arrowhead" orient="auto-start-reverse" markerWidth="6" markerHeight="6" refX="5" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" />
            </marker>
        </defs>

        <!-- Inputs -->
        <text x="70" y="330" class="tf-text">Inputs</text>
        <path class="tf-arrow" d="M70,315 L70,285"/>
        <rect x="20" y="270" width="100" height="30" class="tf-sub-block" />
        <text x="70" y="285" class="tf-text">Input Embeddings</text>
        <path class="tf-arrow" d="M70,270 L70,255"/>
        <rect x="20" y="240" width="100" height="30" class="tf-sub-block" />
        <text x="70" y="255" class="tf-text">Positional Encoding</text>
        
        <!-- Encoder Stack -->
        <rect x="150" y="80" width="120" height="200" class="tf-block"/>
        <text x="210" y="65" class="tf-text" style="font-weight:bold;">Encoder (Nx)</text>
        <path class="tf-arrow" d="M70,240 L70,225 C70,210 140,210 150,210 L150,210"/> 
        
        <!-- Encoder Internal (Simplified) -->
        <rect x="160" y="100" width="100" height="40" class="tf-sub-block"/>
        <text x="210" y="120" class="tf-text">Multi-Head Attention</text>
        <rect x="160" y="150" width="100" height="30" class="tf-sub-block"/>
        <text x="210" y="165" class="tf-text">Add & Norm</text>
        <rect x="160" y="190" width="100" height="40" class="tf-sub-block"/>
        <text x="210" y="210" class="tf-text">Feed Forward</text>
        <rect x="160" y="240" width="100" height="30" class="tf-sub-block"/>
        <text x="210" y="255" class="tf-text">Add & Norm</text>
        <path class="tf-arrow" d="M210,140 L210,150"/>
        <path class="tf-arrow" d="M210,180 L210,190"/>
        <path class="tf-arrow" d="M210,230 L210,240"/>
        
        <!-- Connection Encoder to Decoder -->
        <path class="tf-arrow" d="M270,180 L330,180"/>
        
        <!-- Decoder Stack -->
        <rect x="330" y="80" width="120" height="250" class="tf-block"/>
        <text x="390" y="65" class="tf-text" style="font-weight:bold;">Decoder (Nx)</text>
        
        <!-- Decoder Internal (Simplified) -->
        <rect x="340" y="90
width="100" height="40" class="tf-sub-block"/>
        <text x="390" y="110" class="tf-text">Masked Multi-Head Attention</text>
        <rect x="340" y="140" width="100" height="30" class="tf-sub-block"/>
        <text x="390" y="155" class="tf-text">Add & Norm</text>
        <rect x="340" y="180" width="100" height="40" class="tf-sub-block"/>
        <text x="390" y="200" class="tf-text">Multi-Head Attention</text>
        <text x="390" y="215" class="tf-text tf-text-sm">(Encoder Output)</text>
        <rect x="340" y="230" width="100" height="30" class="tf-sub-block"/>
        <text x="390" y="245" class="tf-text">Add & Norm</text>
        <rect x="340" y="270" width="100" height="30" class="tf-sub-block"/>
        <text x="390" y="285" class="tf-text">Feed Forward</text>
        
        <path class="tf-arrow" d="M390,130 L390,140"/>
        <path class="tf-arrow" d="M390,170 L390,180"/>
        <path class="tf-arrow" d="M390,220 L390,230"/>
        <path class="tf-arrow" d="M390,260 L390,270"/>

        <!-- Outputs -->
        <text x="530" y="330" class="tf-text">Outputs</text>
        <path class="tf-arrow" d="M530,315 L530,225"/>
        <rect x="480" y="210" width="100" height="30" class="tf-sub-block" />
        <text x="530" y="225" class="tf-text">Output Embeddings</text>
        <path class="tf-arrow" d="M530,210 L530,195"/>
        <rect x="480" y="180" width="100" height="30" class="tf-sub-block" />
        <text x="530" y="195" class="tf-text">Positional Encoding</text>
        <path class="tf-arrow" d="M450,110 L450,130 C450,150 460,165 480,180"/> <!-- Path from decoder to output embeddings -->


        <!-- Final Layers -->
        <path class="tf-arrow" d="M390,300 L390,315"/>
        <rect x="340" y="315" width="100" height="25" class="tf-sub-block"/>
        <text x="390" y="327.5" class="tf-text">Linear</text>
        <path class="tf-arrow" d="M390,340 L390,350"/> <!-- Pointing downwards out of view -->
        <text x="390" y="350" class="tf-text" style="opacity:0">Softmax</text> <!-- Placeholder for final output -->
        
        <text x="300" y="20" class="tf-text" style="font-size:16px; font-weight:bold;">Transformer Architecture</text>
    </svg>
    `;
}


