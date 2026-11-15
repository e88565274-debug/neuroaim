package com.jirnyak.neuroaim;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import org.lwjgl.glfw.GLFW;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

public class NeuroAimMod implements ClientModInitializer {
    private static KeyBinding holdKey;
    private static KeyBinding learnKey;
    private static boolean learnOn = true;

    @Override
    public void onInitializeClient() {
        holdKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.neuroaim.hold", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_V, "category.neuroaim"));
        learnKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.neuroaim.learn", InputUtil.Type.KEYSYM, GLFW.GLFW_KEY_K, "category.neuroaim"));

        NeuroAimLogic.init();

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (learnKey.wasPressed()) learnOn = !learnOn;
            NeuroAimLogic.tick(client, holdKey.isPressed(), learnOn);
        });
        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> NeuroAimLogic.save());
    }
}

final class Mlp {
    final int nIn, nHid, nOut;
    final double[][] w1, w2;
    final double[] b1, b2;
    private final double[] h, y, dOut, dH;

    Mlp(int nIn, int nHid, int nOut, long seed) {
        this.nIn = nIn;
        this.nHid = nHid;
        this.nOut = nOut;
        w1 = new double[nHid][nIn];
        w2 = new double[nOut][nHid];
        b1 = new double[nHid];
        b2 = new double[nOut];
        h = new double[nHid];
        y = new double[nOut];
        dOut = new double[nOut];
        dH = new double[nHid];
        Random r = new Random(seed);
        double s1 = 1.0 / Math.sqrt(nIn), s2 = 1.0 / Math.sqrt(nHid);
        for (double[] row : w1) for (int i = 0; i < nIn; i++) row[i] = (r.nextDouble() * 2 - 1) * s1;
        for (double[] row : w2) for (int j = 0; j < nHid; j++) row[j] = (r.nextDouble() * 2 - 1) * s2;
    }

    double[] forward(double[] x) {
        for (int j = 0; j < nHid; j++) {
            double s = b1[j];
            for (int i = 0; i < nIn; i++) s += w1[j][i] * x[i];
            h[j] = Math.tanh(s);
        }
        for (int k = 0; k < nOut; k++) {
            double s = b2[k];
            for (int j = 0; j < nHid; j++) s += w2[k][j] * h[j];
            y[k] = Math.tanh(s);
        }
        return y;
    }

    void train(double[] x, double[] t, double lr) {
        forward(x);
        for (int k = 0; k < nOut; k++) dOut[k] = (y[k] - t[k]) * (1 - y[k] * y[k]);
        for (int j = 0; j < nHid; j++) {
            double s = 0;
            for (int k = 0; k < nOut; k++) s += w2[k][j] * dOut[k];
            dH[j] = s * (1 - h[j] * h[j]);
        }
        for (int k = 0; k < nOut; k++) {
            for (int j = 0; j < nHid; j++) w2[k][j] -= lr * dOut[k] * h[j];
            b2[k] -= lr * dOut[k];
        }
        for (int j = 0; j < nHid; j++) {
            for (int i = 0; i < nIn; i++) w1[j][i] -= lr * dH[j] * x[i];
            b1[j] -= lr * dH[j];
        }
    }

    void save(Path p) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append(nIn).append(' ').append(nHid).append(' ').append(nOut).append('\n');
        for (double[] row : w1) for (double v : row) sb.append(v).append('\n');
        for (double v : b1) sb.append(v).append('\n');
        for (double[] row : w2) for (double v : row) sb.append(v).append('\n');
        for (double v : b2) sb.append(v).append('\n');
        Files.createDirectories(p.getParent());
        Files.writeString(p, sb.toString());
    }

    static Mlp load(Path p) throws IOException {
        List<String> lines = Files.readAllLines(p);
        String[] hdr = lines.get(0).trim().split("\\s+");
        Mlp m = new Mlp(Integer.parseInt(hdr[0]), Integer.parseInt(hdr[1]), Integer.parseInt(hdr[2]), 0L);
        int idx = 1;
        for (double[] row : m.w1) for (int i = 0; i < row.length; i++) row[i] = Double.parseDouble(lines.get(idx++));
        for (int j = 0; j < m.b1.length; j++) m.b1[j] = Double.parseDouble(lines.get(idx++));
        for (double[] row : m.w2) for (int j = 0; j < row.length; j++) row[j] = Double.parseDouble(lines.get(idx++));
        for (int k = 0; k < m.b2.length; k++) m.b2[k] = Double.parseDouble(lines.get(idx++));
        return m;
    }
}

final class NeuroAimLogic {
    static final float MAX_STEP = 10f;
    static final double STRENGTH = 1.0;
    static final double FOV = 90.0;
    static final double RANGE = 4.5;
    static final float NOISE = 0.08f;
    static final float HESITATE = 0.02f;
    static final int REACT_MIN = 3, REACT_MAX = 8;
    static final double ONLINE_LR = 0.004;
    static final int IN = 7, HID = 16, OUT = 2;

    private static final Random RNG = new Random();
    private static Mlp net;
    private static Path weightsPath;
    private static boolean dirty;
    private static int saveTimer;

    private static PlayerEntity locked;
    private static int lockAge, reactAt;
    private static double offX, offY, offZ;
    private static boolean lastValid;
    private static float lastYawErr, lastPitchErr;

    private static float prevYaw, prevPitch;
    private static boolean hasPrev;

    private static boolean pendValid;
    private static final double[] pendX = new double[IN];
    private static final double[] xIn = new double[IN];
    private static final double[] tgt = new double[OUT];

    private NeuroAimLogic() {}

    static void init() {
        weightsPath = FabricLoader.getInstance().getConfigDir().resolve("neuroaim").resolve("weights.txt");
        try {
            if (Files.exists(weightsPath)) {
                Mlp m = Mlp.load(weightsPath);
                if (m.nIn == IN && m.nHid == HID && m.nOut == OUT) {
                    net = m;
                    return;
                }
            }
        } catch (IOException | RuntimeException ignored) {
            // corrupt or old file: retrain below
        }
        net = new Mlp(IN, HID, OUT, 42L);
        pretrain(net);
        dirty = true;
        save();
    }

    static void save() {
        if (net == null || weightsPath == null || !dirty) return;
        try {
            net.save(weightsPath);
            dirty = false;
        } catch (IOException ignored) {
            // read-only config dir: weights stay in memory
        }
    }

    static void tick(MinecraftClient client, boolean held, boolean learnOn) {
        ClientPlayerEntity p = client.player;
        if (p == null || client.world == null || client.currentScreen != null) {
            resetAll();
            return;
        }

        float yaw = p.getYaw(), pitch = p.getPitch();
        float momYaw = hasPrev ? MathHelper.wrapDegrees(yaw - prevYaw) : 0f;
        float momPitch = hasPrev ? pitch - prevPitch : 0f;
        prevYaw = yaw;
        prevPitch = pitch;
        hasPrev = true;

        if (pendValid) {
            if (learnOn && !held) {
                tgt[0] = clamp(momYaw / MAX_STEP, -0.95, 0.95);
                tgt[1] = clamp(momPitch / MAX_STEP, -0.95, 0.95);
                net.train(pendX, tgt, ONLINE_LR);
                dirty = true;
            }
            pendValid = false;
        }

        if (!held && !learnOn) {
            resetLock();
            return;
        }

        Vec3d eye = p.getEyePos();
        Vec3d look = p.getRotationVec(1f);
        PlayerEntity t = pickTarget(client, p, eye, look);
        if (t == null) {
            resetLock();
            return;
        }
        if (t != locked) {
            locked = t;
            lockAge = 0;
            lastValid = false;
            reactAt = REACT_MIN + RNG.nextInt(REACT_MAX - REACT_MIN + 1);
            offX = (RNG.nextDouble() - 0.5) * 0.4;
            offY = (RNG.nextDouble() - 0.5) * 0.3;
            offZ = (RNG.nextDouble() - 0.5) * 0.4;
        }
        if (!held) lockAge = 0;

        Vec3d d = aimOf(t).add(offX, offY, offZ).subtract(eye);
        double horiz = Math.sqrt(d.x * d.x + d.z * d.z);
        float tYaw = (float) Math.toDegrees(Math.atan2(-d.x, d.z));
        float tPitch = (float) -Math.toDegrees(Math.atan2(d.y, horiz));
        float yawErr = MathHelper.wrapDegrees(tYaw - yaw);
        float pitchErr = tPitch - pitch;
        double dist = p.distanceTo(t);

        xIn[0] = clamp(yawErr / 90.0, -1, 1);
        xIn[1] = clamp(pitchErr / 90.0, -1, 1);
        xIn[2] = lastValid ? clamp((yawErr - lastYawErr) / 10.0, -1, 1) : 0;
        xIn[3] = lastValid ? clamp((pitchErr - lastPitchErr) / 10.0, -1, 1) : 0;
        xIn[4] = clamp(dist / RANGE, 0, 1);
        xIn[5] = clamp(momYaw / 10.0, -1, 1);
        xIn[6] = clamp(momPitch / 10.0, -1, 1);

        if (held) {
            lockAge++;
            if (lockAge >= reactAt && RNG.nextFloat() >= HESITATE) {
                double[] out = net.forward(xIn);
                float sy = (float) (out[0] * MAX_STEP * STRENGTH + RNG.nextGaussian() * NOISE);
                float sp = (float) (out[1] * MAX_STEP * STRENGTH + RNG.nextGaussian() * NOISE);
                p.setYaw(yaw + sy);
                p.setPitch(MathHelper.clamp(pitch + sp, -90f, 90f));
            }
        } else if (Math.abs(yawErr) < 30 && Math.abs(pitchErr) < 20 && dist <= RANGE) {
            System.arraycopy(xIn, 0, pendX, 0, IN);
            pendValid = true;
        }

        lastYawErr = yawErr;
        lastPitchErr = pitchErr;
        lastValid = true;

        if (dirty && ++saveTimer >= 1200) {
            saveTimer = 0;
            save();
        }
    }

    private static PlayerEntity pickTarget(MinecraftClient client, ClientPlayerEntity p, Vec3d eye, Vec3d look) {
        if (locked != null && score(p, locked, eye, look) >= Math.cos(Math.toRadians(FOV * 0.6))) {
            return locked;
        }
        double bestDot = Math.cos(Math.toRadians(FOV * 0.5));
        PlayerEntity best = null;
        for (PlayerEntity e : client.world.getPlayers()) {
            double s = score(p, e, eye, look);
            if (s > bestDot) {
                bestDot = s;
                best = e;
            }
        }
        return best;
    }

    private static double score(ClientPlayerEntity p, PlayerEntity e, Vec3d eye, Vec3d look) {
        if (e == p || !e.isAlive() || e.isSpectator() || p.isTeammate(e)) return -2;
        if (p.distanceTo(e) > RANGE || !p.canSee(e)) return -2;
        return look.dotProduct(aimOf(e).subtract(eye).normalize());
    }

    private static Vec3d aimOf(PlayerEntity e) {
        return e.getPos().add(0, e.getHeight() * 0.6, 0);
    }

    private static void pretrain(Mlp m) {
        final int n = 40000;
        double[] x = new double[IN];
        double[] t = new double[OUT];
        for (int it = 0; it < n; it++) {
            double lr = 0.05 * (1.0 - (double) it / n) + 0.002;
            for (int i = 0; i < IN; i++) x[i] = RNG.nextDouble() * 2 - 1;
            t[0] = teacher(x[0] * 90.0, x[2] * 10.0);
            t[1] = teacher(x[1] * 90.0, x[3] * 10.0);
            m.train(x, t, lr);
        }
    }

    private static double teacher(double errDeg, double rateDeg) {
        double u = (0.22 * errDeg + 0.10 * rateDeg) / MAX_STEP + RNG.nextGaussian() * 0.02;
        return clamp(u, -0.95, 0.95);
    }

    private static void resetLock() {
        locked = null;
        lockAge = 0;
        lastValid = false;
    }

    private static void resetAll() {
        resetLock();
        pendValid = false;
        hasPrev = false;
    }

    private static double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
            }
