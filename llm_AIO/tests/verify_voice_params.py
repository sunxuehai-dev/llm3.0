import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.registry import ModelRegistry

def verify_voices():
    print("Verifying CosyVoice Model Parameters...")
    
    models = {m["id"]: m for m in ModelRegistry.get_all_models()}
    
    # 1. Verify CosyVoice v3 Flash
    if "cosyvoice-v3-flash" in models:
        m = models["cosyvoice-v3-flash"]
        voice_param = next((p for p in m["parameters"] if p["name"] == "voice"), None)
        if voice_param:
            count = len(voice_param["options"])
            print(f"✅ cosyvoice-v3-flash: Found 'voice' param with {count} options.")
            if count < 50:
                 print("   ⚠️ WARNING: Expected ~70 voices, found fewer.")
            else:
                 print("   Look good.")
        else:
            print("❌ cosyvoice-v3-flash: 'voice' parameter NOT found.")
    else:
        print("❌ cosyvoice-v3-flash model not found.")

    # 2. Verify CosyVoice v3 Plus
    if "cosyvoice-v3-plus" in models:
        m = models["cosyvoice-v3-plus"]
        voice_param = next((p for p in m["parameters"] if p["name"] == "voice"), None)
        if voice_param:
            count = len(voice_param["options"])
            print(f"✅ cosyvoice-v3-plus: Found 'voice' param with {count} options.")
            if count != 2:
                 print(f"   ⚠️ WARNING: Expected 2 voices, found {count}.")
            else:
                 print("   Look good.")
        else:
            print("❌ cosyvoice-v3-plus: 'voice' parameter NOT found.")
    else:
        print("❌ cosyvoice-v3-plus model not found.")

    # 3. Verify CosyVoice v2
    if "cosyvoice-v2" in models:
        m = models["cosyvoice-v2"]
        voice_param = next((p for p in m["parameters"] if p["name"] == "voice"), None)
        if voice_param:
            count = len(voice_param["options"])
            print(f"✅ cosyvoice-v2: Found 'voice' param with {count} options.")
            if count < 100:
                 print("   ⚠️ WARNING: Expected >100 voices, found fewer.")
            else:
                 print("   Look good.")
        else:
            print("❌ cosyvoice-v2: 'voice' parameter NOT found.")
    else:
        print("❌ cosyvoice-v2 model not found.")

if __name__ == "__main__":
    verify_voices()
